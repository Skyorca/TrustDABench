from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class AttackWorkspace:
    def __init__(self, output_dir: Path, new_id: str, source_id: str, attack_type: str):
        self.output_dir = output_dir
        self.new_id = new_id
        self.source_id = source_id
        self.attack_type = attack_type
        self.root = output_dir / "workspaces" / new_id
        self.original = self.root / "original"
        self.work = self.root / "work"
        self.final = self.root / "final"
        self.scratch = self.root / "scratch"
        self.manifest_path = self.root / "manifest.json"

    def prepare(self, source_files: Iterable[Path]) -> None:
        for directory in [self.original, self.work, self.final, self.scratch]:
            directory.mkdir(parents=True, exist_ok=True)
        for source in source_files:
            if not source.exists():
                continue
            # work is an empty intermediate directory. Copying every source
            # into original, work and final tripled I/O for no functional gain.
            for target_dir in [self.original, self.final]:
                shutil.copy2(source, target_dir / source.name)
    
    def mapping(
        self,
        stage: str = "construct",
    ) -> Dict[str, Path]:
        if stage == "validate":
            data_dir = self.final
        else:
            data_dir = self.original

        return {
            "/mnt/original": self.original,
            "/mnt/data": data_dir,
            "/mnt/work": self.work,
            "/mnt/output": self.final,
            "/mnt/scratch": self.scratch,
        }

    def virtual_original_files(
        self,
    ) -> List[str]:
        return [
            f"/mnt/original/{path.name}"
            for path in sorted(
                self.original.iterdir()
            )
            if path.is_file()
        ]

    def virtual_files(self, stage: str = "construct", file_names: Optional[List[str]] = None) -> List[str]:
        base = self.final if stage == "validate" else self.original
        names = file_names if file_names is not None else [p.name for p in sorted(base.iterdir()) if p.is_file()]
        return [f"/mnt/data/{name}" for name in names]

    def final_file_names(self) -> List[str]:
        return [p.name for p in sorted(self.final.iterdir()) if p.is_file()]

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
        for name in [part.strip() for part in input_file.replace(";", "\n").replace("；", "\n").splitlines() if part.strip()]:
            if not (self.final / Path(name).name).exists():
                missing.append(name)
        return missing

    def write_manifest(
        self,
        input_file: str,
        modified: bool,
        edit_summary: str,
        original_files: List[str],
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "id": self.new_id,
            "source_id": self.source_id,
            "attack_type": self.attack_type,
            "original_files": original_files,
            "final_files": self.final_file_names(),
            "file_root": str(self.final.resolve()),
            "input_file": input_file,
            "modified": bool(modified),
            "edit_summary": edit_summary,
            "virtual_paths": {
                "/mnt/original": "workspace/original",
                "/mnt/data": "workspace/final",
                "/mnt/work": "workspace/work",
                "/mnt/output": "workspace/final",
                "/mnt/scratch": "workspace/scratch",
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
