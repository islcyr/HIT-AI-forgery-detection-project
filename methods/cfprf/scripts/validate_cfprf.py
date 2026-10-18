"""Validate real weights and compare the adapter to the official evaluator."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
from audio_forgery import CFPRFDetector
from audio_forgery.cfprf import refine_proposals
from evaluate_CFPRF import Inference_CFPRF


class CachedFDN(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, wave):
        return self.outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    wave, rate = sf.read(ROOT / 'examples/audio/speech.wav', dtype='float32')
    records = []
    for variant in ('HAD', 'PS', 'LAVDF'):
        start = time.perf_counter()
        detector = CFPRFDetector(variant=variant, device=args.device)
        load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = detector.predict(wave, rate, include_frames=True)
        predict_seconds = time.perf_counter() - start
        device_tag = 'gpu' if detector.device.type == 'cuda' else 'cpu'
        destination = ROOT / 'results' / f'cfprf_{variant.lower()}_{device_tag}.json'
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        tensor = torch.from_numpy(wave)[None].to(detector.device)
        with torch.inference_mode():
            outputs = detector.fdn(tensor)
            frame_count = outputs[0].shape[1]
            # Force known coarse regions ONLY to exercise the trained PRN and
            # compare its decoding with the original evaluator. This is not a
            # forged-audio quality test and is not used in model predictions.
            forced = torch.zeros_like(outputs[0])
            forced[:, :, 1] = 10
            forced[:, 0:1, :] = torch.tensor([10., 0.], device=detector.device)
            forced[:, 5:15, :] = torch.tensor([10., 0.], device=detector.device)
            forced[:, 25:45, :] = torch.tensor([10., 0.], device=detector.device)
            from libs.tool import segscore2proposal, proposal_func
            coarse = segscore2proposal(forced[0].cpu().numpy(), proposal_func)[1]
            decoded = refine_proposals(detector.prn, outputs[3], coarse)
            batch = [(tensor, ['sample'], torch.zeros(1, frame_count, device=detector.device))]
            *_, reference = Inference_CFPRF(batch, CachedFDN((forced, *outputs[1:])), detector.prn, 20, detector.device)
            expected = np.asarray(reference['sample']).reshape(-1, 3)
            actual = decoded.copy()
            actual[:, 1:] *= 0.02
            if len(expected):
                expected = expected[np.argsort(expected[:, 1])]
                actual = actual[np.argsort(actual[:, 1])]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            assert len(coarse) == 3
        records.append({
            'variant':variant, 'strict_weight_loading':True,
            'load_seconds':load_seconds, 'predict_seconds':predict_seconds,
            'speech_prediction':result['label'], 'speech_segments':len(result['segments']),
            'forced_coarse_regions':len(coarse),
            'trained_prn_decode_matches_official':True,
            'forced_region_test_is_quality_evaluation':False,
        })
        del detector, tensor, outputs, forced, batch, decoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {'device':args.device,'torch':torch.__version__, 'variants':records}
    (ROOT/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
