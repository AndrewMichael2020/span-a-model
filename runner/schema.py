"""
Output schema for Phi-4-RV inference results.

Designed to match media-organizer's AI extraction schema so results
can be imported directly into the app's PostgreSQL database.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ExtractionResult:
    """Structured output from one image inference call."""

    asset_id: str

    # Text extracted verbatim from the image (OCR)
    ocr_text: Optional[str] = None

    # 2-3 sentence natural-language description
    summary: Optional[str] = None

    # Descriptive keywords for search/filter
    tags: list[str] = field(default_factory=list)

    # Physical objects detected in the image
    objects: list[str] = field(default_factory=list)

    # Location hints: landmarks, signs, geography
    place_clues: Optional[str] = None

    # Broad scene category
    scene: Optional[str] = None

    # Additional noteworthy details
    image_notes: Optional[str] = None

    # Raw model output before structured parsing (for debugging)
    ai_raw: Optional[str] = None

    # Inference metadata
    model: str = "phi4-reasoning-vision-15b-awq"
    thinking_mode: str = "nothink"  # "nothink" | "think"

    # Set if inference failed for this asset
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def error_result(cls, asset_id: str, error: str) -> "ExtractionResult":
        return cls(asset_id=asset_id, error=error)


@dataclass
class JobManifest:
    """Written to GCS before the VM starts; tracks batch progress."""

    job_id: str
    images: list[ImageEntry]
    result_bucket: str
    result_prefix: str          # e.g. "job_abc123/results/"
    image_prefix: str           # e.g. "job_abc123/images/"
    thinking_mode: str = "nothink"
    callback_url: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["images"] = [asdict(img) for img in self.images]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "JobManifest":
        images = [ImageEntry(**img) for img in d.pop("images", [])]
        return cls(images=images, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ImageEntry:
    asset_id: str
    gcs_path: str               # gs://bucket/prefix/asset_id.jpg
    thinking_mode: str = "nothink"
    original_path: Optional[str] = None  # local path on Mac (for reference only)


@dataclass
class JobProgress:
    """Written to GCS as the runner processes images; enables resume on preemption."""

    job_id: str
    total: int
    processed: list[str] = field(default_factory=list)   # asset_ids done
    failed: list[str] = field(default_factory=list)       # asset_ids errored
    completed: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def pending_count(self) -> int:
        return self.total - len(self.processed) - len(self.failed)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "JobProgress":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
