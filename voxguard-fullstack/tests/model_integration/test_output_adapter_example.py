import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from model_integration.make_output_adapter_example import make_output_adapter_example
from model_integration.output_adapter_example import MockTupleNetwork, TupleOutputAdapter
from model_integration.torch_adapter import TorchAdapter
from voxguard.adapters import AudioInput
from voxguard.schemas import Descriptor, validate_prediction


class FixedTupleNetwork(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, waveform):
        return self.output


def test_known_tuple_heads_are_converted_without_replacing_the_utterance_head():
    frame_logits = torch.tensor([[-2.0, 0.0, 2.0], [2.0, -2.0, 0.0]])
    utterance_logits = torch.tensor([[3.0], [-3.0]])
    auxiliary = {"unrelated_features": torch.arange(50)}
    model = TupleOutputAdapter(FixedTupleNetwork((frame_logits, utterance_logits, auxiliary)))
    output = model(torch.ones(2, 4000))
    assert output.keys() == {"global_score", "frame_scores"}
    assert output["frame_scores"].shape == (2, 3)
    assert output["global_score"].shape == (2,)
    assert output["frame_scores"][0].tolist() == pytest.approx([0.880797, 0.5, 0.119203])
    assert output["global_score"].tolist() == pytest.approx([0.047426, 0.952574], abs=1e-6)
    assert output["global_score"][0] != output["frame_scores"][0].max()
    assert model.is_mock is True


@pytest.mark.parametrize("bad_output", [
    {"frames": torch.zeros(1, 3)},
    (torch.zeros(1, 3), torch.zeros(1, 1)),
    (torch.zeros(1, 2), torch.zeros(1, 1), {}),
    (torch.zeros(1, 3, 1), torch.zeros(1, 1), {}),
    (torch.zeros(1, 3), torch.zeros(1), {}),
    (torch.zeros(1, 3), torch.zeros(1, 2), {}),
    (torch.zeros(1, 3, dtype=torch.int64), torch.zeros(1, 1), {}),
    (torch.tensor([[0.0, float("nan"), 0.0]]), torch.zeros(1, 1), {}),
    (torch.zeros(1, 3), torch.tensor([[float("inf")]]), {}),
])
def test_wrapper_rejects_wrong_shapes_missing_tail_and_nonfinite_logits(bad_output):
    adapter = TupleOutputAdapter(FixedTupleNetwork(bad_output))
    with pytest.raises(ValueError):
        adapter(torch.ones(1, 4000))


@pytest.mark.parametrize("waveform", [
    torch.ones(1600), torch.ones(1, 1, 1600), torch.ones(1, 0),
    torch.ones(1, 1600, dtype=torch.int64), torch.tensor([[float("nan")]]),
])
def test_wrapper_rejects_invalid_inputs(waveform):
    with pytest.raises(ValueError):
        TupleOutputAdapter()(waveform)


def test_mock_network_measures_last_partial_frame_without_padding_bias():
    network = MockTupleNetwork()
    wave = torch.zeros(1, 10400)  # 0.65 s at 16 kHz: six full frames and 0.05 s.
    wave[:, 9600:] = 1.0
    frame_logits, utterance_logits, auxiliary = network(wave)
    assert frame_logits.shape == (1, 7)
    assert auxiliary["valid_frame_samples"].tolist() == [1600] * 6 + [800]
    assert frame_logits[0, -1].item() == pytest.approx(1.6)
    assert auxiliary["frame_energy"][0, -1].item() == pytest.approx(1.0)
    assert utterance_logits.item() == pytest.approx(3.0 / 13.0 - 0.3)


@pytest.fixture
def config_path(tmp_path):
    return make_output_adapter_example(tmp_path / "output-example")


def test_generated_checkpoint_loads_on_cpu_and_covers_the_full_original_audio(config_path):
    config = json.loads(config_path.read_text())
    assert config["is_mock"] is True
    assert config["model_id"] == "example-output-adapter"
    assert config["device"] == "cpu"
    adapter = TorchAdapter(config_path)
    assert not adapter.model.training
    assert all(parameter.device.type == "cpu" for parameter in adapter.model.parameters())
    assert adapter.describe()["is_mock"] is True
    assert adapter.describe()["version"].startswith("example-output-adapter-v1-cfg")
    waveform = np.zeros((1, 10400), dtype=np.float32)
    waveform[:, 9600:] = 1.0
    source = AudioInput(waveform, 16000, .65, "output-example-test")
    prediction = adapter.predict(source, "detect_and_localize", True)
    result, _ = validate_prediction(prediction.result, Descriptor.model_validate(adapter.describe()), source.duration_sec, "detect_and_localize")
    bins = result["timeline"]
    assert len(bins) == 7
    assert bins[0]["start_sec"] == 0
    assert bins[-1]["start_sec"] == pytest.approx(.6)
    assert bins[-1]["end_sec"] == pytest.approx(.65)
    assert bins[-1]["spoof_score"] == pytest.approx(0.1679816)
    assert bins[-1]["spoof_score"] != bins[-2]["spoof_score"]
    assert result["decision"]["spoof_score"] == pytest.approx(1.0 - torch.sigmoid(torch.tensor(3.0 / 13.0 - .3)).item())
    assert result["decision"]["score_type"] == "uncalibrated"
    assert len(result["localization"]["segments"]) == 1
    assert result["localization"]["segments"][0]["start_sec"] == 0
    assert result["localization"]["segments"][0]["end_sec"] == pytest.approx(.6)
    detection_only = adapter.predict(source, "detect", False).result
    assert detection_only["localization"]["status"] == "not_requested"
    assert detection_only["timeline"] is None


def test_generated_wrapper_checkpoint_requires_all_prefixed_parameters(config_path):
    config = json.loads(config_path.read_text())
    weights = config_path.parent / config["weights"]
    state = torch.load(weights, map_location="cpu", weights_only=True)
    assert set(state["state_dict"]) == {"network.frame_scale", "network.frame_bias", "network.utterance_scale", "network.utterance_bias"}
    del state["state_dict"]["network.utterance_bias"]
    torch.save(state, weights)
    config["checkpoint_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="Missing key"):
        TorchAdapter(config_path)


def test_generator_refuses_to_overwrite_existing_weights(config_path):
    config_before = config_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        make_output_adapter_example(config_path.parent)
    assert config_path.read_bytes() == config_before
