from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    raw: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def detection(self) -> Dict[str, Any]:
        return self.raw.get("detection", {})

    @property
    def ingestion(self) -> Dict[str, Any]:
        return self.raw.get("ingestion", {})

    @property
    def kinematics(self) -> Dict[str, Any]:
        return self.raw.get("kinematics", {})


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProjectConfig(raw=raw)


def ensure_output_dirs(cfg: ProjectConfig) -> None:
    output_root = Path(cfg.paths.get("output_root", "./outputs"))
    (output_root / "detection" / "frames").mkdir(parents=True, exist_ok=True)
    (output_root / "detection" / "crops").mkdir(parents=True, exist_ok=True)
    (output_root / "detection" / "annotated").mkdir(parents=True, exist_ok=True)
    (output_root / "kinematics").mkdir(parents=True, exist_ok=True)
    (output_root / "synthesis").mkdir(parents=True, exist_ok=True)
