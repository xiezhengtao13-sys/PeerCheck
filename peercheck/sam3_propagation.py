"""Equations 4-6: official SAM 3 tracker training and video propagation.

The upstream library and pretrained checkpoint must be installed separately.
Only annotated keyframes are read during fitting; inference reads no dense GT.
"""
from collections import OrderedDict
from pathlib import Path
import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def build_tracker(checkpoint):
    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(checkpoint_path=str(checkpoint), load_from_HF=False,
                                  compile=False, apply_temporal_disambiguation=True, device='cuda')
    tracker = model.tracker
    # The demo predictor enters a process-wide autocast context in __init__.
    # Balance it: inference and training forward use explicit contexts below,
    # while backward must execute outside autocast (also during recomputation).
    tracker.bf16_context.__exit__(None,None,None)
    tracker.backbone = model.detector.backbone
    tracker.teacher_force_obj_scores_for_mem = False
    tracker.prob_to_dropout_spatial_mem = 0.
    tracker.rng = np.random.default_rng(0)
    assert tracker.image_size == 1008 and tracker.num_maskmem == 7
    assert tracker.max_cond_frames_in_attn == 4 and tracker.use_memory_selection
    return tracker.eval()

def enable_activation_checkpointing(tracker):
    from torch.utils.checkpoint import checkpoint
    paths=['transformer.encoder.layers','sam_mask_decoder.transformer.layers','maskmem_backbone.fuser.layers']
    wrapped=[]
    for path in paths:
        for index,layer in enumerate(tracker.get_submodule(path)):
            if getattr(layer,'_paper3_checkpointed',False):continue
            original=layer.forward
            def make_forward(function,module):
                def forward(*args,**kwargs):
                    if module.training and torch.is_grad_enabled():
                        # Encoder layers are called with CUDA tensors only in
                        # kwargs. Bundle them as positional checkpoint inputs
                        # so PyTorch detects CUDA and preserves its dropout RNG.
                        def bundled(positional,keyword):return function(*positional,**keyword)
                        return checkpoint(bundled,args,kwargs,use_reentrant=False,preserve_rng_state=True)
                    return function(*args,**kwargs)
                return forward
            layer.forward=make_forward(original,layer)
            layer._paper3_checkpointed=True;wrapped.append(f'{path}.{index}')
    return wrapped

def trainable_parameters(tracker,activation_checkpointing=True):
    for name, parameter in tracker.named_parameters():
        parameter.requires_grad_(not name.startswith('backbone.') and not name.startswith('sam_prompt_encoder.point_embeddings.'))
    tracker.train()
    tracker.backbone.eval()
    # SAM3 supplies a training-only GT-presence branch. Without it, predicted
    # absence replaces logits with a constant and blocks segmentation gradients.
    # Presence comes only from labeled training keyframes; eval ignores this flag.
    tracker.teacher_force_obj_scores_for_mem = True
    checkpointed=enable_activation_checkpointing(tracker) if activation_checkpointing else []
    groups = {}
    for name, parameter in tracker.named_parameters():
        if parameter.requires_grad:
            prefix = name.split('.')[0]
            groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    return {'total_trainable': sum(groups.values()), 'modules': groups,
            'frozen_backbone': True, 'frozen_point_embeddings': True,
            'activation_checkpointing_modules':checkpointed,'activation_checkpointing_rng_preserved':True,
            'trainable_names': [n for n,p in tracker.named_parameters() if p.requires_grad]}

class FrozenFeatures:
    """Cache unprojected backbone outputs; decoder conv_s0/conv_s1 stay trainable."""
    def __init__(self, tracker, max_items=160):
        self.tracker = tracker
        self.cache = OrderedDict()
        self.max_items = max_items
        self.hits = self.misses = 0

    def get(self, row):
        key = row['source_image']
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        with Image.open(key) as opened:
            rgb = np.array(ImageOps.exif_transpose(opened).convert('RGB').resize((1008,1008), Image.Resampling.BILINEAR))
        image = torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None] / 255.
        mean = torch.tensor([.485,.456,.406])[None,:,None,None]
        std = torch.tensor([.229,.224,.225])[None,:,None,None]
        image = (image - mean) / std
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.tracker.backbone.forward_image(image.cuda())['sam2_backbone_out']
        cached = {'image': image, 'backbone_fpn': [x.detach().cpu() for x in result['backbone_fpn']],
                  'vision_pos_enc': [x.detach().cpu() for x in result['vision_pos_enc']]}
        self.cache[key] = cached
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return cached

    def prepare(self, row, objects=4):
        cached = self.get(row)
        dtype=cached['backbone_fpn'][0].dtype if torch.is_autocast_enabled('cuda') else next(self.tracker.sam_mask_decoder.parameters()).dtype
        fpn = [x.to(device='cuda',dtype=dtype) for x in cached['backbone_fpn']]
        fpn[0] = self.tracker.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.tracker.sam_mask_decoder.conv_s1(fpn[1])
        output = {'backbone_fpn': fpn, 'vision_pos_enc': [x.to(device='cuda',dtype=dtype) for x in cached['vision_pos_enc']]}
        _, feats, pos, sizes = self.tracker._prepare_backbone_features(output)
        return cached['image'].cuda().expand(objects,-1,-1,-1), [x.expand(-1,objects,-1) for x in feats], [x.expand(-1,objects,-1) for x in pos], sizes

def masks_for_row(row):
    with Image.open(row['label_mask']) as opened:
        semantic = np.array(opened.resize((1008,1008), Image.Resampling.NEAREST))
    return torch.from_numpy(np.stack([semantic == c for c in range(1,5)]))[:,None].cuda()

def forward_clip(tracker, features, rows, condition_frames, need_loss=True):
    assert len(rows) >= 2, 'single-frame point-prompt override must never be reached'
    assert 1 <= len(condition_frames) < len(rows)
    assert 0 in condition_frames
    outputs = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
    predictions, losses, targets = {}, [], []
    for index in list(condition_frames) + [i for i in range(len(rows)) if i not in condition_frames]:
        image, feats, pos, sizes = features.prepare(rows[index])
        truth = masks_for_row(rows[index])
        captured = []
        original = tracker._forward_sam_heads
        def capture(*args, **kwargs):
            if tracker.training and tracker.teacher_force_obj_scores_for_mem:
                kwargs.setdefault('gt_masks', truth)
            result = original(*args, **kwargs)
            captured.append(result)
            return result
        tracker._forward_sam_heads = capture
        try:
            current = tracker.track_step(frame_idx=index, is_init_cond_frame=index in condition_frames,
                current_vision_feats=feats, current_vision_pos_embeds=pos, feat_sizes=sizes, image=image,
                point_inputs=None, mask_inputs=truth if index in condition_frames else None,
                output_dict=outputs, num_frames=len(rows))
        finally:
            tracker._forward_sam_heads = original
        outputs['cond_frame_outputs' if index in condition_frames else 'non_cond_frame_outputs'][index] = current
        predictions[index] = current
        if need_loss and index not in condition_frames:
            assert len(captured) == 1
            sam = captured[0]
            losses.append({'multistep_pred_multimasks_high_res': [sam[1]], 'multistep_pred_ious': [sam[2]],
                           'multistep_object_score_logits': [sam[6]]})
            targets.append(truth[:,0])
    return predictions, losses, torch.stack(targets) if targets else None

def save_tracker(tracker, path, **metadata):
    names = {name for name,p in tracker.named_parameters() if p.requires_grad}
    # Include all tracker state outside the frozen backbone, not just parameters
    # with a nonzero gradient, to preserve buffers and permit exact 0-step reloads.
    state = {name: tensor.detach().cpu() for name,tensor in tracker.state_dict().items() if not name.startswith('backbone.')}
    torch.save({'tracker':state, 'metadata':metadata, 'trainable_names':sorted(names)}, path)

def load_tracker(tracker, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['tracker']
    missing, unexpected = tracker.load_state_dict(state, strict=False)
    assert not unexpected and all(name.startswith('backbone.') for name in missing)
    return checkpoint.get('metadata', {})


def segmentation_loss(outputs, targets, config):
    """Single-device focal + Dice + IoU regression + object-presence loss.

    Four objects include absent classes. The best focal/Dice mask is supervised;
    IoU regression covers all mask candidates. Absent objects contribute only
    their presence loss. Normalization matches the paper training adapter.
    """
    terms = []
    for out, target in zip(outputs, targets):
        z = out['multistep_pred_multimasks_high_res'][0].float()
        quality = out['multistep_pred_ious'][0].float()
        presence = out['multistep_object_score_logits'][0].float()
        y = target[:, None].float().expand_as(z)
        probability = z.sigmoid()
        pt = probability*y+(1-probability)*(1-y)
        alpha = config['focal_alpha']*y+(1-config['focal_alpha'])*(1-y)
        focal = (F.binary_cross_entropy_with_logits(z,y,reduction='none')*(1-pt)**config['focal_gamma']*alpha).flatten(2).mean(-1)
        dice = 1-(2*(probability*y).flatten(2).sum(-1)+1)/(probability.flatten(2).sum(-1)+y.flatten(2).sum(-1)+1)
        foreground = y[:,0].flatten(1).any(1).float()
        best = (config['loss_mask']*focal+config['loss_dice']*dice).argmin(1)
        idx = torch.arange(len(best),device=z.device)
        pred_binary, truth_binary = z>0, y>0
        actual = (pred_binary & truth_binary).flatten(2).sum(-1).float()/(pred_binary | truth_binary).flatten(2).sum(-1).clamp_min(1)
        quality_loss = (quality-actual).abs().mean(1)
        spatial = (config['loss_mask']*focal[idx,best]+config['loss_dice']*dice[idx,best]+config['loss_iou']*quality_loss)*foreground
        object_loss = F.binary_cross_entropy_with_logits(presence.flatten(),foreground)
        terms.append(spatial.mean()+config['loss_class']*object_loss)
    return torch.stack(terms).mean()


def fine_tune(tracker, frames, config, checkpoint, seed=0):
    """Uniform sequence sampling, then contiguous sparse-keyframe clips."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trainable_parameters(tracker)
    tracker.rng = np.random.default_rng(seed)
    no_decay = set()
    for prefix, module in tracker.named_modules():
        if isinstance(module,torch.nn.LayerNorm):
            no_decay.update(f'{prefix}.{name}' for name,_ in module.named_parameters(recurse=False))
    groups = [[],[]]
    for name,p in tracker.named_parameters():
        if p.requires_grad:
            groups[int(name.endswith('bias') or name in no_decay)].append(p)
    optimizer = torch.optim.AdamW([{'params':groups[0],'weight_decay':config['weight_decay']},{'params':groups[1],'weight_decay':0.}],lr=config['learning_rate'])
    sequences = []
    for sequence in sorted({r['sequence'] for r in frames}):
        keys = [{'source_image':str(r['image']),'label_mask':str(r['mask'])} for r in frames if r['sequence']==sequence and r['mask'] is not None]
        if len(keys)>=2:
            sequences.append(keys)
    if not sequences:
        raise ValueError('SAM fitting needs at least two keyframes in a sequence.')
    features = FrozenFeatures(tracker)
    steps = config['optimizer_steps']
    for step in range(steps):
        group = sequences[int(rng.integers(len(sequences)))]
        length = min(config['clip_frames'],len(group))
        start = int(rng.integers(len(group)-length+1))
        clip = group[start:start+length]
        count = int(rng.integers(1,min(2,length-1)+1))
        conditions = [0]+([int(rng.integers(1,length))] if count==2 else [])
        rate = config['learning_rate']*(config['final_lr_fraction']+(1-config['final_lr_fraction'])*(1+math.cos(math.pi*step/steps))/2)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            predictions,outputs,targets = forward_clip(tracker,features,clip,conditions)
            loss = segmentation_loss(outputs,targets,config)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite SAM loss.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in tracker.parameters() if p.requires_grad],config['gradient_clip_norm'])
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite SAM gradients.')
        optimizer.step()
        if step%25==0:
            print(f'SAM step {step+1}/{steps}: loss={loss.item():.6f}',flush=True)
        del predictions,outputs,targets,loss
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    save_tracker(tracker,checkpoint,optimizer_steps=steps,seed=seed)
    tracker.teacher_force_obj_scores_for_mem = False
    return tracker.eval()


def propagate(tracker, frames, config, work, output):
    """Add all four class masks on all keyframes before bidirectional propagation."""
    from .io import save_mask, read_mask
    work, output = Path(work), Path(output)
    records = []
    tracker.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for sequence in sorted({r['sequence'] for r in frames}):
            rows = [r for r in frames if r['sequence']==sequence]
            keys = [i for i,r in enumerate(rows) if r['mask'] is not None]
            if not keys:
                raise ValueError(f'Sequence {sequence} has no keyframe masks.')
            folder = work/sequence
            folder.mkdir(parents=True,exist_ok=True)
            for i,row in enumerate(rows):
                with Image.open(row['image']) as im:
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    rgb.thumbnail((config['working_frame_long_side'],)*2,Image.Resampling.LANCZOS)
                    rgb.save(folder/f'{i:05d}.jpg',quality=95)
            state = tracker.init_state(video_path=str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True,async_loading_frames=False)
            for i in keys:
                mask = read_mask(rows[i]['mask'])
                for c in range(1,5):
                    tracker.add_new_mask(state,frame_idx=i,obj_id=c,mask=torch.from_numpy(mask==c))
            seen = set()
            def accept(index, ids, masks, scores):
                row = rows[int(index)]
                with Image.open(row['image']) as im:
                    width,height = im.size
                native = F.interpolate(masks.float(),size=(height,width),mode='bilinear',align_corners=False)[:,0]
                best,winner = native.max(0)
                semantic = torch.tensor(ids,device=winner.device,dtype=torch.uint8)[winner]
                semantic[best<=0] = 0
                p = scores.float().sigmoid().flatten()
                appearing = (native>0).flatten(1).any(1)
                confidence = torch.where(appearing,p,1-p).mean().item()
                save_mask(output/sequence/(Path(row['frame_id']).stem+'.png'),semantic.cpu().numpy())
                records.append({'frame_id':row['frame_id'],'sam_conf':confidence})
                seen.add(int(index))
                print(f'SAM propagation {sequence}: {len(seen)}/{len(rows)}',flush=True)
            first = min(keys)
            for i,ids,_,masks,scores in tracker.propagate_in_video(state,start_frame_idx=first,max_frame_num_to_track=len(rows),reverse=False,tqdm_disable=True,propagate_preflight=True):
                accept(i,ids,masks,scores)
            if first>0:
                for i,ids,_,masks,scores in tracker.propagate_in_video(state,start_frame_idx=first,max_frame_num_to_track=first,reverse=True,tqdm_disable=True,propagate_preflight=False):
                    if int(i)<first:
                        accept(i,ids,masks,scores)
            if len(seen)!=len(rows):
                raise RuntimeError('Propagation did not cover every frame.')
            del state
            gc.collect()
            torch.cuda.empty_cache()
    return records
