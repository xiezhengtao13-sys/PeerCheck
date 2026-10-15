import json
import math
import re
import struct

import pytest

from peercheck import release


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def test_released_files_match_manifest():
    assert 'uniform_rho5_r0' in release.available()
    for setting in release.available():
        release.verify(setting)


def test_released_tensors():
    folder = release.setting_dir('uniform_rho5_r0')
    decoder = header(folder / 'dino_decoder.safetensors')
    metadata = [decoder.pop('__metadata__', {})]
    trainable = sum(math.prod(v['shape']) for k, v in decoder.items() if not re.search('running_|num_batches', k))
    assert len(decoder) == 25 and trainable == 2_056_197
    index = json.loads((folder / 'index.json').read_text())
    sam = {}
    for shard in set(index['weight_map'].values()):
        h = header(folder / shard)
        metadata.append(h.pop('__metadata__', {}))
        sam.update(h)
    assert set(sam) == set(index['weight_map'])
    assert sum(math.prod(v['shape']) for v in sam.values()) == index['metadata']['trainable_parameters']
    assert not any(k.startswith(release.FROZEN_PREFIXES) for k in sam)
    assert not re.search(r'[A-Za-z]:\\|/Users/|/home/', json.dumps(metadata))


def test_decoder_loads_strictly():
    pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    head = release.load_decoder('uniform_rho5_r0', device='cpu')
    assert sum(p.numel() for p in head.parameters()) == 2_056_197
