"""Pipeline timing logging utilities."""


def print_pipeline_timing(endpoint: str, durations: dict[str, float]) -> None:
    """Print formatted execution timings for ingestion pipeline phases."""
    print(f"\n=== {endpoint} Pipeline Timing ===")
    print(f"CSV Ingestion: {durations['csv_ingestion']:.2f}s")
    print(f"Data Cleaning: {durations['data_cleaning']:.2f}s")
    print(f"Data Profiling: {durations['data_profiling']:.2f}s")
    print(f"LLM Summary Generation: {durations['llm_summary']:.2f}s")
    print(f"Total Pipeline: {durations['total']:.2f}s")
    print("=" * (len(endpoint) + 20))
