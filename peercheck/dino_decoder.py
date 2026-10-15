"""Equations 7-8: frozen DINOv2 ViT-L/14 and the paper RGB-skip decoder."""
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """2,056,197 trainable parameters; 72x96 -> 576x768 for paper inputs."""
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(1024, 256, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(259, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
            for _ in range(3)])
        self.classifier = nn.Conv2d(256, 5, 1)

    def forward(self, features, rgb):
        x = self.project(features)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            resized = F.interpolate(rgb, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, resized], 1))
        return self.classifier(x)


def normalized_rgb(rgb):
    mean = rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std = rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return (rgb-mean)/std


def load_encoder(config, device):
    from transformers import Dinov2Model
    source = os.environ.get('PEERCHECK_DINO_SOURCE', config['model'])
    kwargs = {'attn_implementation': 'sdpa'}
    if not Path(source).is_dir():
        kwargs['revision'] = config['revision']
    model = Dinov2Model.from_pretrained(source, **kwargs).eval().to(device)
    model.requires_grad_(False)
    return model


def extract_features(frames, config, cache, device='cuda'):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(config, device)
    width, height = config['input_wh']
    if width % 14 or height % 14:
        raise ValueError('DINO input dimensions must be multiples of 14.')
    with torch.inference_mode():
        for i, row in enumerate(frames):
            with Image.open(row['image']) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((width,height),Image.Resampling.BICUBIC)).copy()
            x = torch.from_numpy(rgb).permute(2,0,1)[None].to(device).float()/255
            tokens = encoder(pixel_values=normalized_rgb(x)).last_hidden_state[:,1:]
            feature = tokens[0].T.reshape(1024,height//14,width//14).half().cpu().numpy()
            np.save(cache/f'{i:05d}_feature.npy', feature)
            np.save(cache/f'{i:05d}_rgb.npy', rgb)
            print(f'DINO features {i+1}/{len(frames)}', flush=True)
    del encoder
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def train_decoder(frames, config, cache, checkpoint, seed=0, device='cuda'):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache = Path(cache)
    keys = [i for i,r in enumerate(frames) if r['mask'] is not None]
    if not keys:
        raise ValueError('At least one keyframe mask is required.')
    counts = np.zeros(5, dtype=np.int64)
    labels = {}
    for i in keys:
        with Image.open(frames[i]['mask']) as im:
            native = np.asarray(im)
            if not np.isin(native, range(5)).all():
                raise ValueError('Training masks must contain complete labels 0-4.')
            counts += np.bincount(native.ravel(),minlength=5)
            labels[i] = np.asarray(im.resize(tuple(config['output_wh']),Image.Resampling.NEAREST)).copy()
    weights = (np.maximum(counts,1)/counts.sum())**-.5
    weights /= weights.mean()
    ce_weights = torch.tensor(weights,dtype=torch.float32,device=device)
    head = Decoder().to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=tuple(config['adam_betas']))
    order = []
    steps, warmup, batch = config['steps'], config['warmup_steps'], config['batch']
    for step in range(steps):
        while len(order) < batch:
            order.extend(rng.permutation(keys).tolist())
        picked, order = order[:batch], order[batch:]
        features = torch.from_numpy(np.stack([np.load(cache/f'{i:05d}_feature.npy') for i in picked])).to(device).float()
        rgb = torch.from_numpy(np.stack([np.load(cache/f'{i:05d}_rgb.npy') for i in picked])).permute(0,3,1,2).to(device).float()/255
        truth = torch.from_numpy(np.stack([labels[i] for i in picked])).to(device).long()
        rate = config['lr']*(step+1)/warmup if step<warmup else config['lr']*(1+math.cos(math.pi*(step-warmup)/max(steps-warmup,1)))/2
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            logits = head(features,rgb)
        logits = logits.float()
        ce = F.cross_entropy(logits,truth,weight=ce_weights)
        p = logits.softmax(1)
        y = F.one_hot(truth,5).permute(0,3,1,2).float()
        smooth = config['dice_smooth']
        dice = 1-((2*(p*y).sum((0,2,3))+smooth)/(p.sum((0,2,3))+y.sum((0,2,3))+smooth)).mean()
        loss = config['ce_coefficient']*ce + config['dice_coefficient']*dice
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite decoder loss.')
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step+1==steps:
            print(f'DINO step {step+1}/{steps}: loss={loss.item():.6f}',flush=True)
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'head':head.state_dict(),'steps':steps,'class_weights':weights.tolist()},checkpoint)
    return head.eval()


def infer(frames, head, cache, output, device='cuda'):
    from .io import save_mask
    cache, output = Path(cache), Path(output)
    records = []
    head.eval()
    with torch.inference_mode():
        for i,row in enumerate(frames):
            f = torch.from_numpy(np.load(cache/f'{i:05d}_feature.npy')).unsqueeze(0).to(device).float()
            rgb = torch.from_numpy(np.load(cache/f'{i:05d}_rgb.npy')).permute(2,0,1)[None].to(device).float()/255
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                logits = head(f,rgb)
            with Image.open(row['image']) as im:
                width,height = im.size
            native = F.interpolate(logits.float(),size=(height,width),mode='bilinear',align_corners=False)
            mask = native.argmax(1)[0].byte().cpu().numpy()
            confidence = native.softmax(1).max(1).values.mean().item()
            save_mask(output/row['sequence']/(Path(row['frame_id']).stem+'.png'),mask)
            records.append({'frame_id':row['frame_id'],'dino_conf':confidence})
    return records
