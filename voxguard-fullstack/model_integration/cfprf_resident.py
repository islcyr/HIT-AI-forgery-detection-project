"""FastAPI-side adapter: preprocess audio, call resident CFPRF, convert output."""
import json
import os
from pathlib import Path
import struct
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np
from voxguard.adapters import AdapterError
from voxguard.audio import resample_mono
from voxguard.schemas import Descriptor, validate_prediction
from .cfprf_output import convert_output


class ResidentAdapter:
    def __init__(self, config, endpoint):
        self.config = config
        self.descriptor = Descriptor.model_validate(config['descriptor'])
        self.endpoint = endpoint.rstrip('/')
        if not self.endpoint.startswith('http://127.0.0.1:'):
            raise ValueError('Resident model endpoint must use loopback')
        self.client = build_opener(ProxyHandler({}))
        health = self._request('/health', timeout=10)
        if not health.get('ready') or health.get('descriptor') != self.describe():
            raise ValueError('Resident model is not ready or its version differs from configuration')

    def describe(self):
        return self.descriptor.model_dump()

    def _request(self, path, body=None, timeout=540):
        request = Request(self.endpoint + path, data=body, headers={'Content-Type':'audio/wav'} if body is not None else {})
        try:
            with self.client.open(request, timeout=timeout) as response:
                return json.loads(response.read(8 * 1024 * 1024))
        except HTTPError as error:
            raise AdapterError('MODEL_SERVICE_ERROR', f'模型服务返回 HTTP {error.code}，请查看模型日志。', error.code == 503) from error
        except (URLError, TimeoutError, OSError) as error:
            raise AdapterError('MODEL_UNAVAILABLE', '常驻模型服务不可用或响应超时，请检查启动进程。', True) from error

    def predict(self, audio, mode, include_timeline):
        rate = self.descriptor.input.sample_rate_hz
        mono = resample_mono(audio, rate)
        data = np.asarray(mono, dtype='<f4').tobytes()
        wav = (b'RIFF' + struct.pack('<I',36 + len(data)) + b'WAVEfmt ' +
               struct.pack('<IHHIIHH',16,3,1,rate,rate*4,4,32) + b'data' + struct.pack('<I',len(data)) + data)
        response = self._request('/predict', wav)
        try:
            context = {**self.describe(), 'checkpoint_variant':self.config['checkpoint_variant']}
            prediction = convert_output(response['result'], audio, context, mode)
            validate_prediction(prediction.result, self.descriptor, audio.duration_sec, mode)
            return prediction
        except (ValueError, KeyError, TypeError) as error:
            raise AdapterError('INVALID_MODEL_OUTPUT', 'CFPRF 输出不符合后端约定，请查看任务日志。') from error


def create_adapter():
    path = Path(os.environ['VOXGUARD_MODEL_CONFIG']).resolve()
    return ResidentAdapter(json.loads(path.read_text()), os.getenv('VOXGUARD_MODEL_ENDPOINT','http://127.0.0.1:8765'))
