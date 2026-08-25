from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class AttackWorkspace:
    def __init__(self, output_dir: Path, new_id: str, source_id: str, attack_type: str):
        self.output_dir = output_dir.resolve()
        self.new_id = new_id
        self.source_id = source_id
        self.attack_type = attack_type
        self.container = self.output_dir / "workspaces" / new_id
        self.active_pointer = self.container / "active.json"
        self._judge_snapshot: Optional[Path] = None
        self._set_root(self._load_active_root() or self.container)

    def prepare(self, source_files: Iterable[Path]) -> None:
        # Never recycle an earlier construct directory.  A timed-out tool can
        # leave an NFS .nfs* handle or a Windows workbook handle behind.  A new
        # generation makes retries independent of that abandoned process.
        self._activate_new_generation()
        for directory in [self.original, self.work, self.final, self.scratch]:
            directory.mkdir(parents=True, exist_ok=True)
        for source in source_files:
            if not source.exists():
                continue
            for target_dir in [self.original, self.work, self.final]:
                shutil.copy2(source, target_dir / source.name)

    def _set_root(self, root: Path) -> None:
        self.root = root
        self.original = self.root / "original"
        self.work = self.root / "work"
        self.final = self.root / "final"
        self.scratch = self.root / "scratch"
        self.manifest_path = self.root / "manifest.json"
        self._judge_snapshot = None

    def _load_active_root(self) -> Optional[Path]:
        try:
            payload = json.loads(self.active_pointer.read_text(encoding="utf-8"))
            generation = str(payload.get("generation", "")).strip()
            candidate = self.container / generation
            if generation and candidate.is_dir():
                return candidate
        except (OSError, ValueError, TypeError):
            pass
        # Read legacy flat workspaces so existing checkpoints remain usable.
        if (self.container / "manifest.json").exists():
            return self.container
        return None

    def _activate_new_generation(self) -> None:
        self.container.mkdir(parents=True, exist_ok=True)
        generation = f"run-{time.time_ns()}"
        root = self.container / generation
        root.mkdir(parents=False, exist_ok=False)
        self._set_root(root)
        payload = {"generation": generation, "updated_at_ns": time.time_ns()}
        temporary = self.container / f".active-{time.time_ns()}.json"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(self.active_pointer)

    def mapping(self, stage: str = "construct") -> Dict[str, Path]:
        if stage == "judge":
            if self._judge_snapshot is None:
                raise RuntimeError("judge snapshot has not been prepared")
            return {
                "/mnt/original": self._judge_snapshot / "original",
                "/mnt/data": self._judge_snapshot / "attacked",
                "/mnt/scratch": self._judge_snapshot / "scratch",
            }
        if stage == "validate":
            return {
                "/mnt/original": self.original,
                "/mnt/data": self.final,
                "/mnt/work": self.work,
                "/mnt/output": self.final,
                "/mnt/scratch": self.scratch,
            }
        return {
            "/mnt/data": self.original,
            "/mnt/work": self.work,
            "/mnt/output": self.final,
            "/mnt/scratch": self.scratch,
        }

    def prepare_judge_snapshot(self, attempt: int) -> Path:
        """Give a semantic judge a disposable, read-only-in-practice evidence view.

        The agent receives no virtual route to ``final``. Tool code is still run
        locally, so copying both packages prevents accidental writes from affecting
        the accepted candidate even if a judge ignores its prompt.
        """
        judge_root = self.root / "judge" / f"attempt-{attempt}-{time.time_ns()}"
        original = judge_root / "original"
        attacked = judge_root / "attacked"
        scratch = judge_root / "scratch"
        for directory in (original, attacked, scratch):
            directory.mkdir(parents=True, exist_ok=True)
        for source in self.original.iterdir():
            if source.is_file():
                shutil.copy2(source, original / source.name)
        for source in self.final.iterdir():
            if source.is_file():
                shutil.copy2(source, attacked / source.name)
        self._judge_snapshot = judge_root
        return judge_root

    def tool_scratch(self, stage: str) -> Path:
        if stage == "judge":
            if self._judge_snapshot is None:
                raise RuntimeError("judge snapshot has not been prepared")
            return self._judge_snapshot / "scratch"
        return self.scratch

    def virtual_files(
        self,
        stage: str = "construct",
        file_names: Optional[List[str]] = None,
        original: bool = False,
    ) -> List[str]:
        if stage == "judge":
            if self._judge_snapshot is None:
                raise RuntimeError("judge snapshot has not been prepared")
            base = self._judge_snapshot / ("original" if original else "attacked")
            virtual_root = "/mnt/original" if original else "/mnt/data"
        elif original:
            base = self.original
            virtual_root = "/mnt/original" if stage == "validate" else "/mnt/data"
        elif stage == "validate":
            base = self.final
            virtual_root = "/mnt/data"
        else:
            base = self.original
            virtual_root = "/mnt/data"
        names = file_names if file_names is not None else [p.name for p in sorted(base.iterdir()) if p.is_file()]
        return [f"{virtual_root}/{Path(name).name}" for name in names]

    def final_file_names(self) -> List[str]:
        return [p.name for p in sorted(self.final.iterdir()) if p.is_file()]

    def original_file_names(self) -> List[str]:
        return [p.name for p in sorted(self.original.iterdir()) if p.is_file()]

    def normalize_final_files(self, output_files: Optional[List[str]]) -> None:
        if not output_files:
            return
        wanted = {Path(name).name for name in output_files}
        for path in list(self.final.iterdir()):
            if path.is_file() and path.name not in wanted:
                path.unlink()
        for name in wanted:
            target = self.final / name
            if target.exists():
                continue
            for source_dir in [self.work, self.original]:
                source = source_dir / name
                if source.exists():
                    shutil.copy2(source, target)
                    break

    def validate_file_refs(self, input_file: str) -> List[str]:
        missing: List[str] = []
        for name in _split_names(input_file):
            if not (self.final / Path(name).name).exists():
                missing.append(name)
        return missing

    def file_diff(self) -> Dict[str, Any]:
        original = _inventory(self.original)
        final = _inventory(self.final)
        original_names = set(original)
        final_names = set(final)
        common = original_names & final_names
        return {
            "added": sorted(final_names - original_names),
            "removed": sorted(original_names - final_names),
            "changed": sorted(name for name in common if original[name]["sha256"] != final[name]["sha256"]),
            "unchanged": sorted(name for name in common if original[name]["sha256"] == final[name]["sha256"]),
            "original": original,
            "final": final,
        }

    def write_manifest(
        self,
        input_file: str,
        modified: bool,
        edit_summary: str,
        original_files: List[str],
        transformation_record: Dict[str, Any],
        integrity_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "id": self.new_id,
            "source_id": self.source_id,
            "attack_type": self.attack_type,
            "original_files": original_files,
            "final_files": self.final_file_names(),
            "file_root": str(self.final.absolute()),
            "file_root_relative": str(self.final),
            "input_file": input_file,
            "modified": bool(modified),
            "edit_summary": edit_summary,
            "transformation_record": transformation_record,
            "integrity_report": integrity_report,
            "file_diff": self.file_diff(),
            "virtual_paths": {
                "construct:/mnt/data": "workspace/original",
                "construct:/mnt/output": "workspace/final",
                "validate:/mnt/original": "workspace/original",
                "validate:/mnt/data": "workspace/final",
                "judge:/mnt/original": "workspace/judge/<attempt>/original",
                "judge:/mnt/data": "workspace/judge/<attempt>/attacked",
                "judge:/mnt/scratch": "workspace/judge/<attempt>/scratch",
                "/mnt/work": "workspace/work",
                "/mnt/scratch": "workspace/scratch",
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _split_names(value: str) -> List[str]:
    return [part.strip() for part in value.replace(";", "\n").replace("；", "\n").splitlines() if part.strip()]


def _inventory(directory: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def _rmtree_retry(path: Path, attempts: int = 5, delay_sec: float = 0.2) -> None:
    last_error: Optional[Exception] = None
    for index in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if index == attempts - 1:
                break
            time.sleep(delay_sec * (index + 1))
    if last_error is not None:
        raise last_error


def _retire_workspace(path: Path) -> None:
    """Move stale workspaces out of the active namespace before cleanup.

    A timed-out Python tool can leave a transient .nfs* file. NFS refuses to
    unlink that file while a process still has it open, but it can normally move
    the parent directory. Keeping an unreaped retired directory is preferable
    to aborting a new construct/repair attempt.
    """
    retired_root = path.parent / ".retired"
    retired_root.mkdir(parents=True, exist_ok=True)
    retired = retired_root / f"{path.name}.{time.time_ns()}"
    try:
        path.rename(retired)
    except FileNotFoundError:
        return
    except OSError as exc:
        # Never fall back to recursive deletion here.  On Windows a second
        # process may still hold an .xlsx handle; on NFS a .nfs* file may be
        # held by a timed-out tool.  Deleting the active tree is both unsafe
        # and no more likely to succeed.  The caller gets a clear retryable
        # error instead of an EBUSY/WinError 32 traceback from shutil.rmtree.
        raise OSError(f"workspace is busy and cannot be retired: {path}") from exc
    try:
        _rmtree_retry(retired, attempts=2, delay_sec=0.1)
    except OSError:
        # Best effort only: a later run or manual maintenance may remove it.
        pass
