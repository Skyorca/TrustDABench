from __future__ import annotations

import concurrent.futures
import time
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
        on_stall: Optional[Callable[[List[str], float], None]] = None,
    ) -> List[Any]:
        if self.num_workers == 1:
            results = []
            for row in tqdm(rows, total=len(rows)):
                results.append(process_func(row))
            return results

        results: List[Any] = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
        last_completion = time.monotonic()
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
                        if on_stall is not None:
                            on_stall(
                                pending_ids,
                                time.monotonic() - last_completion,
                            )
                        # A stall is diagnostic only. Individual API/tool/attack
                        # deadlines are responsible for stopping slow work.
                        continue
                    for future in done:
                        results.append(future.result())
                        progress.update(1)
                    last_completion = time.monotonic()
            executor.shutdown(wait=True)
            return results
        except BaseException:
            for future in pending:
                future.cancel()

            executor.shutdown(
                wait=True,
                cancel_futures=True,
            )
            raise
