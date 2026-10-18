"""Deliberately broken or slow adapters for service boundary tests; not models."""
import time
from voxguard.adapters import MockAdapter, Prediction


class BrokenAdapter(MockAdapter):
    def predict(self, audio, mode, include_timeline):
        prediction = super().predict(audio, mode, include_timeline)
        prediction.result["decision"]["spoof_score"] = float("nan")
        return prediction


class SlowAdapter(MockAdapter):
    def predict(self, audio, mode, include_timeline):
        time.sleep(10)
        return super().predict(audio, mode, include_timeline)


def broken_factory():
    return BrokenAdapter()


def slow_factory():
    return SlowAdapter()
