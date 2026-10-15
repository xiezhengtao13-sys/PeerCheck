"""GPU demo: pretrained SAM 3 propagation, fresh DINO decoder, mutual audit.

This two-keyframe demonstration is distinct from the fine-tuned paper run.
"""
import argparse
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sam-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('full_demo_output'))
    p.add_argument('--work',type=Path,default=Path('full_demo_work'))
    p.add_argument('--config',type=Path,default=base.parent/'configs/paper.yaml')
    a = p.parse_args()
    from peercheck.pipeline import run
    from quick_demo import quick_demo
    timing = run(base/'data',a.output,a.work,a.config,a.sam_checkpoint,sam_mode='pretrained',seed=0)
    evaluation = quick_demo(base/'data',a.output,base/'paper_reference.csv',a.output,compare_paper=False)
    print(json.dumps({'execution':timing,'evaluation':evaluation},indent=2))


if __name__=='__main__':
    main()
