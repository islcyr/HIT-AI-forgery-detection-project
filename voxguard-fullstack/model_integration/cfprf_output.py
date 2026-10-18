"""Convert native CFPRF output without inventing frame scores."""
import math

from voxguard.adapters import Prediction

DECISION_RULE = "at_least_one_refined_proposal_with_verification_gt_0.5"


def convert_output(raw, audio, descriptor, mode):
    """Output converter contract: (raw JSON, AudioInput, Descriptor dict, mode)."""
    if raw.get("model") != "CFPRF" or raw.get("checkpoint_variant") != descriptor["checkpoint_variant"]:
        raise ValueError("Unexpected CFPRF model/checkpoint variant")
    if descriptor["thresholds"]["detection"] != 0.5 or descriptor["thresholds"]["localization"] != 0.5 or descriptor["decision_operator"] != "gt":
        raise ValueError("CFPRF converter requires verification > 0.5")
    if raw.get("decision_rule") != DECISION_RULE:
        raise ValueError("Unsupported CFPRF decision rule")
    rate = raw.get("sample_rate")
    if type(rate) is not int or rate != descriptor["input"]["sample_rate_hz"]:
        raise ValueError("Unexpected analysis sample rate")
    duration = raw.get("duration")
    if type(duration) not in (int, float) or not math.isfinite(duration) or not math.isclose(duration, audio.duration_sec, abs_tol=1 / rate, rel_tol=0):
        raise ValueError("CFPRF duration differs from original audio")
    if type(raw.get("is_fake")) is not bool or raw.get("label") != ("fake" if raw["is_fake"] else "real"):
        raise ValueError("Inconsistent CFPRF label/is_fake; expected fake or real")
    if not isinstance(raw.get("score_type"), str) or not raw["score_type"]:
        raise ValueError("Missing native score type")
    segments = raw.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Missing CFPRF segments")
    resolution = raw.get("frame_resolution")
    if resolution != descriptor["localization_resolution_sec"]:
        raise ValueError("Unexpected CFPRF resolution")
    converted = [{"start_sec": float(s["start"]), "end_sec": float(s["end"]), "spoof_score": float(s["score"])} for s in segments]
    # Sorting preserves proposal scores. Overlaps are rejected by the service;
    # combining overlapping proposals requires a separately versioned policy.
    converted.sort(key=lambda s: (s["start_sec"], s["end_sec"]))
    requested = mode == "detect_and_localize"
    return Prediction({
        "decision": {"label": "spoof" if raw["is_fake"] else "bonafide", "spoof_score": float(raw["score"]),
                     "threshold": descriptor["thresholds"]["detection"], "threshold_operator": "gt",
                     "score_type": "uncalibrated", "source_score_type": raw["score_type"],
                     "score_direction": "higher_is_spoof", "decision_rule": raw["decision_rule"]},
        "localization": {"status": "available" if requested else "not_requested", "score_aggregation": "proposal",
                         "segments": converted if requested else None,
                         "resolution_sec": resolution if requested else None,
                         "threshold": descriptor["thresholds"]["localization"] if requested else None,
                         "postprocessing_version": "cfprf-native-proposals-v1" if requested else None},
        "timeline": None,
    }, rate)
