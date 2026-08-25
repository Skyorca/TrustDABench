from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm


class Runner:
    def __init__(self, num_workers: int = 4, stall_timeout_sec: Optional[float] = 600):
        self.num_workers = max(1, int(num_workers))
        if stall_timeout_sec is None or str(stall_timeout_sec).strip().lower() in {"", "none", "null", "0"}:
            self.stall_timeout_sec = None
        else:
            self.stall_timeout_sec = float(stall_timeout_sec)

    def run(
        self,
        rows: List[Dict[str, Any]],
        process_func: Callable[[Dict[str, Any]], Any],
    ) -> List[Any]:
        if self.num_workers == 1:
            results = []
            for row in tqdm(rows, total=len(rows)):
                try:
                    results.append(process_func(row))
                except Exception as exc:
                    sample_id = str(row.get("id", "unknown"))
                    tqdm.write(f"ERROR sample {sample_id} failed without stopping the batch: {exc}")
                    results.append({"id": sample_id, "status": "runner_error", "error": str(exc)})
            return results

        results: List[Any] = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
        try:
            future_to_row = {executor.submit(process_func, row): row for row in rows}
            pending = set(future_to_row)
            with tqdm(total=len(future_to_row)) as progress:
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=self.stall_timeout_sec,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        pending_ids = [str(future_to_row[future].get("id", "unknown")) for future in pending]
                        tqdm.write(
                            "WARNING no complete sample within "
                            f"{self.stall_timeout_sec:g} seconds; workers are still running. "
                            f"Pending sample ids: {pending_ids}"
                        )
                        continue
                    for future in done:
                        row = future_to_row[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            sample_id = str(row.get("id", "unknown"))
                            tqdm.write(f"ERROR sample {sample_id} failed without stopping the batch: {exc}")
                            results.append({"id": sample_id, "status": "runner_error", "error": str(exc)})
                        progress.update(1)
            executor.shutdown(wait=True)
            return results
        except Exception:
            for future in future_to_row:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
