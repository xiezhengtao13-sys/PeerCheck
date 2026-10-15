"""Select nested temporal or appearance keyframes from image sequences."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from peercheck.io import discover
from peercheck.keyframes import select, appearance_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Directory containing images/<sequence>/<frame>.jpg')
    parser.add_argument('--output',type=Path,required=True,help='Output keyframe CSV')
    parser.add_argument('--rho',type=float,default=5,help='Percent of all frames to annotate; at least one per sequence')
    parser.add_argument('--strategy',choices=['uniform','coverage-aware'],default='uniform')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/paper.yaml')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    frames = discover(args.input)
    config = yaml.safe_load(args.config.read_text())
    embeddings = appearance_embeddings(frames,config['dino'],args.device) if args.strategy=='coverage-aware' else None
    selected = select(frames,args.rho,args.strategy,args.seed,embeddings)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{k:frames[i][k] for k in ['sequence','frame_id','frame_index']} for i in selected]).to_csv(args.output,index=False)
    print(f'Selected {len(selected)} of {len(frames)} frames.')


if __name__=='__main__':
    main()
