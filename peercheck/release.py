"""Released paper models: DINOv2 decoder and fine-tuned SAM 3 tracker parameters.

Each setting folder under weights/ holds the decoder state (MIT) and the trained
SAM 3 tracker parameters (SAM License). The SAM 3 file contains only the
parameters trained in the paper; the frozen image backbone, point embeddings and
buffers come from the official SAM 3 checkpoint, which must be obtained separately.
"""
import hashlib
import json
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'
FROZEN_PREFIXES = ('backbone.', 'sam_prompt_encoder.point_embeddings.')


def available():
    return sorted(p.name for p in WEIGHTS.iterdir() if (p / 'manifest.json').is_file()) if WEIGHTS.is_dir() else []


def setting_dir(setting):
    folder = WEIGHTS / setting
    if not (folder / 'manifest.json').is_file():
        raise ValueError(f'Unknown release {setting!r}; available: {", ".join(available()) or "none"}.')
    return folder


def verify(setting):
    """Check every released file against the SHA-256 recorded in manifest.json."""
    folder = setting_dir(setting)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        data = (folder / name).read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'{setting}/{name} does not match manifest.json.')
    return manifest


def load_decoder(setting, device='cuda'):
    """Return the paper's DINOv2 decoder for `setting`, in eval mode."""
    from safetensors.torch import load_file
    from .dino_decoder import Decoder
    head = Decoder()
    head.load_state_dict(load_file(setting_dir(setting) / 'dino_decoder.safetensors'), strict=True)
    return head.to(device).eval()


def load_sam(tracker, setting):
    """Overwrite the trained parameters of an official SAM 3 tracker in place.

    `tracker` comes from sam3_propagation.build_tracker with the official
    checkpoint. The released tensors must cover exactly the parameters trained
    in the paper, with matching shapes.
    """
    import torch
    from safetensors.torch import load_file
    folder = setting_dir(setting)
    index = json.loads((folder / 'index.json').read_text())
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(load_file(folder / shard))
    if set(state) != set(index['weight_map']):
        raise ValueError('Released SAM 3 shards do not match index.json.')
    params = dict(tracker.named_parameters())
    trained = {n for n in params if not n.startswith(FROZEN_PREFIXES)}
    if set(state) != trained:
        raise ValueError(f'Released SAM 3 parameters differ from the trained set: '
                         f'{len(set(state) - trained)} unexpected, {len(trained - set(state))} missing.')
    wrong = [n for n, t in state.items() if params[n].shape != t.shape]
    if wrong:
        raise ValueError(f'Shape mismatch for {len(wrong)} SAM 3 parameters, e.g. {wrong[0]}.')
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(params[name].device, params[name].dtype))
    return sum(t.numel() for t in state.values())
