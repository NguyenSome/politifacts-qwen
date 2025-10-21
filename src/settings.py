from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, Field, AnyUrl
import yaml


class MLflowSettings(BaseModel):
    enable: bool = False
    tracking_uri: Optional[Union[AnyUrl, str]] = None
    exp_name: str = "default"
    run_name: Optional[str] = None
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
    def load_from_yaml(cls, yaml_path: str = "configs/base.yaml") -> "AppSettings":
        """Load settings from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        
        with path.open("r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        
        return cls(**yaml_config)