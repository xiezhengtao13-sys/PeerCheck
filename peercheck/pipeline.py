"""Train on keyframes, propagate, and mutually verify at input resolution."""
from pathlib import Path
import gc
import json
import time
import pandas as pd
import yaml
from .io import discover, read_mask, save_mask, overlay
from .verification import agreement, intersection_labels, route


def run(data, output, work, config_path, sam_checkpoint, sam_mode='finetune', seed=0):
    import torch
    from . import sam3_propagation as sam
    from . import dino_decoder as dino
    if not torch.cuda.is_available():
        raise RuntimeError('Full inference requires a CUDA GPU. Use quick_demo for CPU.')
    output, work = Path(output), Path(work)
    output.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text())
    frames = discover(data)
    for sequence in {r['sequence'] for r in frames}:
        if not any(r['mask'] is not None for r in frames if r['sequence']==sequence):
            raise ValueError(f'No keyframe in {sequence}.')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tracker = sam.build_tracker(sam_checkpoint)
    if sam_mode == 'finetune':
        sam.fine_tune(tracker,frames,config['sam'],work/'sam_tracker_last.pt',seed)
    sam.propagate(tracker,frames,config['sam'],work/'sam_frames',output/'V3')
    sam_seconds = time.perf_counter()-started
    del tracker
    gc.collect()
    torch.cuda.empty_cache()
    dino_start = time.perf_counter()
    dino.extract_features(frames,config['dino'],work/'dino_cache')
    head = dino.train_decoder(frames,config['dino'],work/'dino_cache',work/'dino_last.pt',seed)
    dino.infer(frames,head,work/'dino_cache',output/'VD')
    dino_seconds = time.perf_counter()-dino_start
    records = []
    for row in frames:
        relative = Path(row['sequence'])/(Path(row['frame_id']).stem+'.png')
        s,d = read_mask(output/'V3'/relative),read_mask(output/'VD'/relative)
        ci = agreement(s,d,config['verification']['a_min'])
        intersection = intersection_labels(s,d)
        save_mask(output/'V8'/relative,intersection)
        dest = output/'overlays'/relative.with_suffix('.jpg')
        dest.parent.mkdir(parents=True,exist_ok=True)
        overlay(row['image'],intersection).save(dest,quality=90)
        records.append({'sequence':row['sequence'],'frame_id':row['frame_id'],'C_i':ci,
                        'status':route(ci,config['verification']['tau'],row['mask'] is not None)})
    table = pd.DataFrame(records)
    table.to_csv(output/'decisions.csv',index=False)
    for status in ['accept','review']:
        table[table.status==status].to_csv(output/f'{status}.csv',index=False)
    timing = {'sam_mode':sam_mode,'sam_seconds':sam_seconds,'dino_seconds':dino_seconds,
              'total_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name(0),
              'dino_steps':config['dino']['steps'],'seed':seed,'input_frames':len(frames),
              'keyframes':sum(r['mask'] is not None for r in frames),'dense_GT_used_for_training':False}
    (output/'execution.json').write_text(json.dumps(timing,indent=2)+'\n')
    return timing
