"""Run PeerCheck on your own image sequences and keyframe labels."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Directory containing images/ and keyframes/ sequence folders')
    parser.add_argument('--output',type=Path,required=True,help='Predictions, overlays and decision CSVs')
    parser.add_argument('--work',type=Path,required=True,help='Private scratch directory for features and trained checkpoints')
    parser.add_argument('--sam-checkpoint',type=Path,required=True,help='Authorized official pretrained SAM 3 checkpoint')
    parser.add_argument('--sam-mode',choices=['finetune','pretrained'],default='finetune')
    parser.add_argument('--release',help='Use released paper models (e.g. uniform_rho5_r0) instead of training')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/paper.yaml')
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    from peercheck.pipeline import run
    print(run(args.input,args.output,args.work,args.config,args.sam_checkpoint,args.sam_mode,args.seed,args.release))


if __name__=='__main__':
    main()
