"""Independent enrichment requests share one bounded process-wide pool."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import BoundedSemaphore
from time import monotonic

# Two active dataset jobs share these slots; neither creates its own pool.
_requests = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-request")
_EARLY_REQUEST_SLOTS = BoundedSemaphore(4)


def submit_early_requests(requests):
    """Start at most two early requests per upload in the shared AI pool.

    The semaphore bounds the executor's otherwise unbounded submission queue.
    If capacity is unavailable, callers can use the ordinary post-validation
    path without having partially submitted an early batch.
    """
    requests = list(requests.items())
    if not requests:
        return {}
    if len(requests) > 2:
        raise ValueError("An upload may start at most two early AI requests.")

    acquired = 0
    for _ in requests:
        if not _EARLY_REQUEST_SLOTS.acquire(blocking=False):
            for _ in range(acquired):
                _EARLY_REQUEST_SLOTS.release()
            return {}
        acquired += 1

    futures = {}
    try:
        for name, callback in requests:
            future = _requests.submit(callback)
            futures[name] = future
            future.add_done_callback(lambda _future: _EARLY_REQUEST_SLOTS.release())
    except Exception:
        unsubmitted = len(requests) - len(futures)
        for future in futures.values():
            future.cancel()
        for _ in range(unsubmitted):
            _EARLY_REQUEST_SLOTS.release()
        return {}
    return futures


def run_parallel_enrichment(summary, columns, *, total_columns, publish,
                            is_current, timeout_seconds=60, schema=None, labels=None,
                            schema_after_summary=False, early_futures=None,
                            early_handlers=None):
    if schema is not None and columns:
        raise ValueError('Use schema or column requests, not both')
    early_futures = dict(early_futures or {})
    early_handlers = dict(early_handlers or {})
    if any(name not in {"summary", "labels"} for name in early_futures):
        raise ValueError("Only summary and label requests may be adopted early.")
    defer_schema = bool(schema_after_summary and summary and schema)
    tasks = deque(([('summary', None, summary)] if summary and 'summary' not in early_futures else []) +
                  ([('schema', None, schema)] if schema and not defer_schema else []) +
                  ([('labels', None, labels)] if labels and 'labels' not in early_futures else []) +
                  ([('column', name, callback) for name, callback in columns] if schema is None else []))
    pending = {future: (kind, None) for kind, future in early_futures.items()}
    proposals = {}
    column_results = {}
    label_proposals = {}
    description = None
    completed = 0
    planned = (len(columns) + bool(summary or 'summary' in early_futures)
               + bool(schema) + bool(labels or 'labels' in early_futures))
    schema_coverage = None
    column_roles = {}
    column_labels = {}
    semantic_revision = 0
    labels_completed = False
    deadline = monotonic() + timeout_seconds
    stopped = None
    halt_columns = False

    def merge_labels(proposal, label_proposal):
        """Apply label fields without changing a role proposal's decision."""
        if not isinstance(proposal, dict):
            proposal = {}
        if not isinstance(label_proposal, dict):
            return dict(proposal)
        merged = dict(proposal)
        label_decision = label_proposal.get('decision')
        if isinstance(label_decision, dict) and isinstance(merged.get('decision'), dict):
            decision = dict(merged['decision'])
            for field in ('display_name', 'label_evidence_strength'):
                if field in label_decision:
                    decision[field] = label_decision[field]
            if decision or 'decision' in merged:
                merged['decision'] = decision
        for field, value in label_proposal.items():
            if isinstance(field, str) and field.startswith('label_'):
                merged[field] = value
        return merged

    def merge_label_results():
        for name, label_proposal in label_proposals.items():
            if name not in proposals and isinstance(label_proposal, dict):
                proposals[name] = dict(label_proposal)
            else:
                proposals[name] = merge_labels(proposals.get(name), label_proposal)

    def advance_revision(result):
        nonlocal semantic_revision
        try:
            semantic_revision = max(semantic_revision, int(result.get('semantic_revision', 0)))
        except (AttributeError, TypeError, ValueError):
            pass

    def snapshot():
        failed = sum(item.get('status') in {'error', 'unavailable', 'disabled'}
                     for item in column_results.values() if isinstance(item, dict))
        return {'summary': description, 'interpretation_proposals': dict(proposals),
                'column_roles': dict(column_roles), 'column_labels': dict(column_labels), 'semantic_revision': semantic_revision,
                'coverage': (dict(schema_coverage)
                             if schema_coverage is not None else {'total_columns': total_columns, 'selected_columns': len(columns),
                             'completed_columns': len(column_results), 'failed_columns': failed,
                             'skipped_columns': total_columns - len(columns),
                             'complete': len(column_results) == total_columns and failed == 0,
                             'stop_reason': stopped}),
                'progress': min(99, 10 + int(89 * completed / max(1, planned)))}

    def fill():
        if halt_columns:
            return
        while len(pending) < 4 and is_current() and monotonic() < deadline:
            if not tasks:
                break
            task = tasks.popleft()
            kind, name, callback = task
            def invoke(callback=callback, kind=kind, description=description):
                if not is_current():
                    return None
                return callback(description) if kind == 'schema' and schema_after_summary else callback()
            pending[_requests.submit(invoke)] = (kind, name)

    try:
        if is_current():
            fill()
            if is_current():
                publish(snapshot())
            else:
                stopped = 'stale_version'
        else:
            stopped = 'stale_version'
        while pending and stopped is None:
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
                handler = early_handlers.get(kind) if kind in early_futures else None
                if handler is not None and result is not None:
                    try:
                        result = handler(result)
                    except Exception:
                        # Keep local apply failures private, just like provider failures.
                        result = None
                if kind == 'summary':
                    description = result
                    if defer_schema:
                        # The driver schedules the dependent call after summary
                        # completion, so no shared-pool worker waits on another.
                        tasks.append(('schema', None, schema))
                        defer_schema = False
                elif kind == 'schema':
                    if isinstance(result, dict) and 'interpretation_proposals' in result and 'coverage' in result:
                        proposals = dict(result['interpretation_proposals'])
                        schema_coverage = result['coverage']
                        column_roles = result.get('column_roles', {})
                        if not labels:
                            column_labels = result.get('column_labels', {})
                        advance_revision(result)
                        if labels_completed:
                            merge_label_results()
                    else:
                        schema_coverage = {'total_columns': total_columns, 'selected_columns': 0,
                            'completed_columns': 0, 'failed_columns': 0, 'skipped_columns': total_columns,
                            'complete': False, 'stop_reason': 'schema_unavailable'}
                elif kind == 'labels':
                    labels_completed = True
                    if isinstance(result, dict) and 'interpretation_proposals' in result:
                        label_proposals = result['interpretation_proposals']
                        column_labels = result.get('column_labels', {})
                        advance_revision(result)
                        merge_label_results()
                else:
                    if not isinstance(result, dict) or 'status' not in result:
                        result = {'status': 'error', 'error_code': 'invalid_result'}
                    column_results[name] = result
                    if labels_completed and name in label_proposals:
                        result = merge_labels(result, label_proposals[name])
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
