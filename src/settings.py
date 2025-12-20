from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import AnyUrl, BaseModel, Field


class MLflowSettings(BaseModel):
    enable: bool = False
    tracking_uri: AnyUrl | str | None = None
    exp_name: str = "default"
    run_name: str | None = None
    artifact_path: str = "artifacts"

    def resolved_run_name(self, prefix="finetune") -> str:
        if self.run_name:
            return self.run_name
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{ts}"


class AppSettings(BaseModel):
    seed: int = 13
    data_path: Path = Path("data/train.json")
    output_dir: Path = Path("outputs")
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "configs/base.yaml") -> AppSettings:
        """Load settings from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        return cls(**yaml_config)
