from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm


class Runner:
    def __init__(self, num_workers: int = 4, stall_timeout_sec: Optional[float] = 600):
        self.num_workers = max(1, int(num_workers))
        if stall_timeout_sec is None or str(stall_timeout_sec).strip().lower() in {"", "none", "null", "0"}:
            self.stall_timeout_sec = None
        else:
            self.stall_timeout_sec = float(stall_timeout_sec)

    def run_completed(
        self,
        rows: List[Dict[str, Any]],
        process_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Iterable[Dict[str, Any]]:
        if self.num_workers == 1:
            for row in tqdm(rows, total=len(rows)):
                yield process_func(row)
            return

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
        future_to_row = {executor.submit(process_func, row): row for row in rows}
        pending = set(future_to_row)
        try:
            with tqdm(total=len(future_to_row)) as progress:
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=self.stall_timeout_sec,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        pending_ids = [str(future_to_row[future].get("id", "unknown")) for future in pending]
                        raise TimeoutError(
                            "No sample finished within "
                            f"{self.stall_timeout_sec} seconds. Pending sample ids: {pending_ids}"
                        )
                    for future in done:
                        progress.update(1)
                        yield future.result()
            executor.shutdown(wait=True)
        except Exception:
            for future in future_to_row:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
