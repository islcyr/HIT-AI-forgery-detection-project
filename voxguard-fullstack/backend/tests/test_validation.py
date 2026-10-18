from copy import deepcopy

import numpy as np
import pytest

from voxguard.adapters import AudioInput, MockAdapter
from voxguard.schemas import Descriptor, validate_prediction


def valid_result():
    adapter = MockAdapter()
    audio = AudioInput(np.ones((1, 32000), dtype=np.float32), 16000, 2.0, "fixture")
    return adapter.predict(audio, "detect_and_localize", True).result, Descriptor.model_validate(adapter.describe())


@pytest.mark.parametrize("kind", ["nan", "label", "threshold", "gap", "overlap", "bounds", "mean", "resolution"])
def test_bad_model_values_are_rejected(kind):
    result, descriptor = valid_result()
    if kind == "nan": result["decision"]["spoof_score"] = float("nan")
    if kind == "label": result["decision"]["label"] = "bonafide"
    if kind == "threshold": result["decision"]["threshold"] = 100.0
    if kind == "gap": result["timeline"][1]["start_sec"] += 0.01
    if kind == "overlap": result["timeline"][1]["start_sec"] -= 0.01
    if kind == "bounds": result["localization"]["segments"][0]["end_sec"] = 3.0
    if kind == "mean": result["localization"]["segments"][0]["spoof_score"] = 3.0
    if kind == "resolution": result["localization"]["resolution_sec"] = 0.001
    with pytest.raises(ValueError):
        validate_prediction(result, descriptor, 2.0, "detect_and_localize")


def test_cross_head_disagreement_is_reported_not_rewritten():
    result, descriptor = valid_result()
    result["decision"].update(label="bonafide", spoof_score=-1.0)
    output, warnings = validate_prediction(result, descriptor, 2.0, "detect_and_localize")
    assert output["decision"]["label"] == "bonafide"
    assert output["localization"]["segments"]
    assert warnings[0]["code"] == "GLOBAL_LOCALIZATION_DISAGREE"
