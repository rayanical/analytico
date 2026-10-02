"""Independent enrichment requests share one bounded process-wide pool."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import monotonic

# Two active dataset jobs share these slots; neither creates its own pool.
_requests = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-request")


def run_parallel_enrichment(summary, columns, *, total_columns, publish,
                            is_current, timeout_seconds=60):
    tasks = iter(([('summary', None, summary)] if summary else []) +
                 [('column', name, callback) for name, callback in columns])
    pending = {}
    proposals = {}
    description = None
    completed = 0
    planned = len(columns) + bool(summary)
    deadline = monotonic() + timeout_seconds
    stopped = None
    halt_columns = False

    def snapshot():
        failed = sum(item['status'] in {'error', 'unavailable', 'disabled'} for item in proposals.values())
        return {'summary': description, 'interpretation_proposals': dict(proposals),
                'coverage': {'total_columns': total_columns, 'selected_columns': len(columns),
                             'completed_columns': len(proposals), 'failed_columns': failed,
                             'skipped_columns': total_columns - len(columns),
                             'complete': len(proposals) == total_columns and failed == 0,
                             'stop_reason': stopped},
                'progress': min(99, 10 + int(89 * completed / max(1, planned)))}

    def fill():
        if halt_columns:
            return
        while len(pending) < 4 and is_current() and monotonic() < deadline:
            task = next(tasks, None)
            if task is None:
                break
            kind, name, callback = task
            def invoke(callback=callback):
                return callback() if is_current() else None
            pending[_requests.submit(invoke)] = (kind, name)

    try:
        fill()
        publish(snapshot())
        while pending:
            if not is_current():
                stopped = 'stale_version'
                break
            remaining = deadline - monotonic()
            if remaining <= 0:
                stopped = 'time_budget'
                break
            done, _ = wait(pending, timeout=min(.1, remaining), return_when=FIRST_COMPLETED)
            if not is_current():
                stopped = 'stale_version'
                break
            for future in done:
                kind, name = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    # Never publish provider exception text, prompts or secrets.
                    result = None if kind == 'summary' else {'status': 'error', 'error_code': 'request_failed'}
                if kind == 'summary':
                    description = result
                else:
                    if not isinstance(result, dict) or 'status' not in result:
                        result = {'status': 'error', 'error_code': 'invalid_result'}
                    proposals[name] = result
                    if result.get('error_code') in {'provider_unavailable', 'provider_http_error'} or result['status'] == 'disabled':
                        # Don't fan a bad key or provider outage out across the
                        # rest of a wide file. Retain already-running results.
                        halt_columns = True
                        stopped = 'provider_unavailable'
                completed += 1
            if done:
                publish(snapshot())
                fill()
        if completed < planned and stopped is None:
            stopped = 'stale_version' if not is_current() else 'time_budget'
        return snapshot()
    finally:
        # In-flight HTTP requests retain their own timeout; their eventual
        # results cannot publish after this worker stops. Cancel queued work.
        for future in pending:
            future.cancel()
