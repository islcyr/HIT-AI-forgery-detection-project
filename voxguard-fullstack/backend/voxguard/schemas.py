"""Validate external adapter data before it becomes an API result."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Capabilities(StrictModel):
    detection: bool
    localization: bool


class ModelInput(StrictModel):
    sample_rate_hz: int = Field(gt=0)
    min_duration_sec: float = Field(gt=0)
    max_duration_sec: float = Field(gt=0)


class Thresholds(StrictModel):
    detection: float
    localization: float | None
    version: str = Field(min_length=1)


class Descriptor(StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    decision_operator: Literal["gte", "gt"] = "gte"
    display_name: str = Field(min_length=1)
    status: Literal["ready", "unavailable"]
    is_mock: bool
    capabilities: Capabilities
    input: ModelInput
    localization_resolution_sec: float | None = Field(gt=0)
    thresholds: Thresholds

    @model_validator(mode="after")
    def consistency(self):
        if not self.capabilities.detection:
            raise ValueError("The first API version requires whole-utterance detection")
        if self.input.min_duration_sec > self.input.max_duration_sec:
            raise ValueError("Invalid model duration limits")
        if self.capabilities.localization != (self.localization_resolution_sec is not None):
            raise ValueError("Localization resolution must match capability")
        if self.capabilities.localization != (self.thresholds.localization is not None):
            raise ValueError("Localization threshold must match capability")
        if self.is_mock and not self.id.startswith("example-"):
            raise ValueError("Mock model IDs must start with example-")
        return self


class Decision(StrictModel):
    label: Literal["bonafide", "spoof"]
    spoof_score: float
    threshold: float
    score_type: Literal["uncalibrated"]
    score_direction: Literal["higher_is_spoof"]
    threshold_operator: Literal["gte", "gt"] = "gte"
    source_score_type: str | None = Field(default=None, min_length=1)
    decision_rule: str | None = Field(default=None, min_length=1)


class Interval(StrictModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    spoof_score: float


class Localization(StrictModel):
    status: Literal["available", "unsupported", "not_requested"]
    score_aggregation: Literal["timeline_mean", "proposal"] = "timeline_mean"
    segments: list[Interval] | None
    resolution_sec: float | None
    threshold: float | None
    postprocessing_version: str | None


class Result(StrictModel):
    decision: Decision
    localization: Localization
    timeline: list[Interval] | None


def validate_prediction(raw: dict, descriptor: Descriptor, duration: float, mode: str):
    result = Result.model_validate(raw)
    decision, loc = result.decision, result.localization
    if decision.threshold != descriptor.thresholds.detection:
        raise ValueError("Detection threshold differs from the frozen descriptor")
    if decision.threshold_operator != descriptor.decision_operator:
        raise ValueError("Decision operator differs from descriptor")
    fake = decision.spoof_score > decision.threshold if decision.threshold_operator == "gt" else decision.spoof_score >= decision.threshold
    if decision.label != ("spoof" if fake else "bonafide"):
        raise ValueError("Label and score disagree")
    if mode == "detect":
        expected = "not_requested" if descriptor.capabilities.localization else "unsupported"
        if loc.status != expected or any(x is not None for x in (loc.segments, loc.resolution_sec, loc.threshold, loc.postprocessing_version, result.timeline)):
            raise ValueError("Invalid detection-only localization output")
        return result.model_dump(), []
    if loc.status != "available" or loc.segments is None:
        raise ValueError("Localization must succeed when requested")
    if loc.resolution_sec != descriptor.localization_resolution_sec or loc.threshold != descriptor.thresholds.localization:
        raise ValueError("Localization configuration differs from descriptor")
    if not loc.postprocessing_version:
        raise ValueError("Missing localization postprocessing version")
    if (loc.score_aggregation == "timeline_mean" and not result.timeline) or (result.timeline is not None and len(result.timeline) > 200000) or len(loc.segments) > 10000:
        raise ValueError("Missing or excessive internal timeline")
    previous = 0.0
    for interval in result.timeline or []:
        if interval.start_sec != previous or interval.end_sec <= interval.start_sec or interval.end_sec > duration:
            raise ValueError("Timeline must exactly cover the original duration without gaps or overlap")
        previous = interval.end_sec
    if result.timeline is not None and previous != duration:
        raise ValueError("Timeline does not end at the decoded duration")
    previous = 0.0
    for segment in loc.segments:
        if segment.start_sec < previous or segment.end_sec <= segment.start_sec or segment.end_sec > duration:
            raise ValueError("Invalid segment order or boundary")
        previous = segment.end_sec
        if loc.score_aggregation == "proposal":
            continue
        weighted = sum(max(0.0, min(segment.end_sec, b.end_sec) - max(segment.start_sec, b.start_sec)) * b.spoof_score for b in result.timeline)
        mean = weighted / (segment.end_sec - segment.start_sec)
        if not math.isclose(segment.spoof_score, mean, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Segment score must equal its time-weighted timeline mean")
    warnings = []
    if (decision.label == "spoof") != bool(loc.segments):
        warnings.append({"code": "GLOBAL_LOCALIZATION_DISAGREE", "message": "整段判断与定位结果存在差异，请分别查看。"})
    return result.model_dump(), warnings
