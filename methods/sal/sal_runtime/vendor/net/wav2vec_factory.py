"""s3prl local factory without importing its full hub or Fairseq training code."""
from pathlib import Path
import torch


def wav2vec2_local(ckpt, fairseq=True):
    from s3prl.upstream.wav2vec2.expert import UpstreamExpert
    source = Path(ckpt)
    if not fairseq:
        return UpstreamExpert(str(source))
    converted = source.with_name(source.stem + '.sal-converted.pt')
    if not converted.is_file():
        from filelock import FileLock
        with FileLock(str(converted) + '.lock'):
            if not converted.is_file():
                # Same three fields as s3prl's load_and_convert_fairseq_ckpt.
                # This particular shared XLS-R checkpoint already stores plain
                # dict config, so Fairseq/OmegaConf conversion is unnecessary.
                state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
                cfg = state['cfg']
                if not isinstance(cfg, dict):
                    raise ValueError('Expected the shared XLS-R checkpoint with plain dict cfg.')
                temporary = converted.with_suffix('.tmp')
                torch.save({'task_cfg': cfg['task'], 'model_cfg': cfg['model'],
                            'model_weight': state['model']}, temporary)
                temporary.replace(converted)
    return UpstreamExpert(str(converted))
