from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("VOXGUARD_DATA_DIR", "data")).resolve())
    frontend_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "frontend")
    enable_mock: bool = field(default_factory=lambda: os.getenv("ENABLE_MOCK", "true").lower() == "true")
    adapter_factory: str = field(default_factory=lambda: os.getenv("VOXGUARD_ADAPTER_FACTORY", ""))
    cors_origins: list[str] = field(default_factory=lambda: [x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()])
    max_upload_bytes: int = 20 * 1024 * 1024
    min_duration_sec: float = 0.5
    max_duration_sec: float = 120.0
    queue_capacity: int = field(default_factory=lambda: int(os.getenv("QUEUE_CAPACITY", "4")))
    task_timeout_sec: float = field(default_factory=lambda: float(os.getenv("TASK_TIMEOUT_SEC", "600")))
    model_load_timeout_sec: float = field(default_factory=lambda: float(os.getenv("MODEL_LOAD_TIMEOUT_SEC", "120")))
    retention_sec: float = field(default_factory=lambda: float(os.getenv("RETENTION_SEC", "86400")))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        self.frontend_dir = Path(self.frontend_dir).resolve()
        if self.queue_capacity < 1 or self.task_timeout_sec <= 0 or self.model_load_timeout_sec <= 0 or self.retention_sec <= self.task_timeout_sec:
            raise ValueError("QUEUE_CAPACITY >= 1 and RETENTION_SEC > TASK_TIMEOUT_SEC > 0 are required")
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must list explicit origins, not *")

    def limits(self):
        return {"accepted_formats": ["wav", "flac", "mp3"], "max_upload_bytes": self.max_upload_bytes,
                "min_duration_sec": self.min_duration_sec, "max_duration_sec": self.max_duration_sec}
