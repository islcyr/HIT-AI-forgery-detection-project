"""Existing backend preprocessing, with a single resident ensemble endpoint."""
import json
import os
from pathlib import Path
import struct

import numpy as np
from voxguard.adapters import AdapterError
from voxguard.audio import resample_mono
from voxguard.schemas import validate_prediction
from .cfprf_resident import ResidentAdapter
from .ensemble_output import convert_output


class EnsembleAdapter(ResidentAdapter):
    def predict(self, audio, mode, include_timeline):
        rate = self.descriptor.input.sample_rate_hz
        data = np.asarray(resample_mono(audio, rate), dtype='<f4').tobytes()
        wav = (b'RIFF' + struct.pack('<I', 36+len(data)) + b'WAVEfmt ' +
               struct.pack('<IHHIIHH', 16, 3, 1, rate, rate*4, 4, 32) + b'data' +
               struct.pack('<I', len(data)) + data)
        response = self._request('/predict', wav)
        try:
            prediction = convert_output(response['result'], audio, self.describe(), mode)
            validate_prediction(prediction.result, self.descriptor, audio.duration_sec, mode)
            return prediction
        except (ValueError, KeyError, TypeError) as error:
            raise AdapterError('INVALID_MODEL_OUTPUT', '综合检测结果不符合接口约定，请查看任务日志。') from error


def create_adapter():
    config = json.loads(Path(os.environ['VOXGUARD_MODEL_CONFIG']).resolve().read_text())
    return EnsembleAdapter(config, os.getenv('VOXGUARD_MODEL_ENDPOINT', 'http://127.0.0.1:8765'))
