from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dataset import parse_input_files
from utils import safe_id


class SampleWorkspace:
    def __init__(self, row: Dict[str, Any], keep: bool = False, base_dir: str | Path | None = None):
        self.row = row
        self.keep = bool(keep)
        self.base_dir = Path(base_dir) if base_dir else None
        self.root: Path | None = None
        self.files_dir: Path | None = None
        self.manifest: Dict[str, Any] = {}

    def __enter__(self) -> "SampleWorkspace":
        prefix = safe_id(str(self.row["id"])) + "_"
        if self.keep:
            assert self.base_dir is not None
            self.root = self.base_dir / safe_id(str(self.row["id"]))
            if self.root.exists():
                shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = Path(tempfile.mkdtemp(prefix=prefix))
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._copy_files()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep and self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def _copy_files(self) -> None:
        assert self.files_dir is not None
        source_root = Path(str(self.row["file_root"]))
        copied: List[Dict[str, str]] = []
        for name in parse_input_files(self.row):
            safe_name = Path(name).name
            src = source_root / name
            if not src.exists():
                raise FileNotFoundError(f"Input file not found for {self.row['id']}: {src}")
            dst = self.files_dir / safe_name
            shutil.copy2(src, dst)
            copied.append({"input_name": name, "workspace_name": safe_name, "source_path": str(src)})

        self.manifest = {
            "id": self.row["id"],
            "workspace": str(self.root),
            "files_dir": str(self.files_dir),
            "files": copied,
        }

    @property
    def file_names(self) -> List[str]:
        return [item["workspace_name"] for item in self.manifest.get("files", [])]

    @property
    def working_dir(self) -> str:
        assert self.files_dir is not None
        return str(self.files_dir)
