"""Optional fixed-step YOLO26 segmentation comparator (Ultralytics AGPL)."""
import argparse
import math
import shutil
from pathlib import Path
import numpy as np
import yaml
from peercheck.io import discover, read_mask, save_mask


def prepare(frames, work):
    import cv2
    paths=[]
    for row in frames:
        if row['mask'] is None:
            continue
        image=work/'images'/row['sequence']/row['frame_id']
        image.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(row['image'],image)
        dest=work/'labels'/row['sequence']/(image.stem+'.txt')
        dest.parent.mkdir(parents=True,exist_ok=True)
        mask=read_mask(row['mask']);h,w=mask.shape;lines=[]
        for c in range(1,5):
            count,components,stats,_=cv2.connectedComponentsWithStats((mask==c).astype(np.uint8),connectivity=8)
            for index in range(1,count):
                x,y,cw,ch,area=map(int,stats[index])
                if area<50:
                    continue
                contours,_=cv2.findContours((components[y:y+ch,x:x+cw]==index).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points=contour[:,0].astype(float)+[x,y]
                    if len(points)<3:
                        x0,y0=min(x,w-2),min(y,h-2)
                        x1,y1=min(w-1,x0+max(cw-1,1)),min(h-1,y0+max(ch-1,1))
                        points=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
                    lines.append(str(c-1)+' '+' '.join(f'{v:.10f}' for v in (points/[w,h]).ravel()))
        dest.write_text('\n'.join(lines)+'\n')
        paths.append(str(image.resolve()))
    if not paths:
        raise ValueError('No keyframe labels.')
    train=work/'train.txt';train.write_text('\n'.join(paths)+'\n')
    data=work/'data.yaml'
    data.write_text(yaml.safe_dump({'path':str(work.resolve()),'train':str(train.resolve()),'val':str(train.resolve()),
                                  'names':{0:'girder_and_handrail',1:'deck',2:'pier',3:'inter_girder_region'}}))
    return data,len(paths)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',default='yolo26m-seg.pt')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'configs/paper.yaml')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    cfg=yaml.safe_load(a.config.read_text())['yolo']
    frames=discover(a.input);data,k=prepare(frames,a.work)
    target,batch=cfg['optimizer_steps'],cfg['batch']
    class FixedStepTrainer(SegmentationTrainer):
        def __init__(self,*args,**kwargs):
            self.actual_steps=0
            super().__init__(*args,**kwargs)
        def optimizer_step(self):
            rate=cfg['lr0']*(cfg['final_lr_fraction']+(1-cfg['final_lr_fraction'])*(1+math.cos(math.pi*self.actual_steps/target))/2)
            if self.actual_steps<cfg['warmup_steps']:
                rate*=(self.actual_steps+1)/cfg['warmup_steps']
            for group in self.optimizer.param_groups:
                group['lr']=rate
            super().optimizer_step()
            self.actual_steps+=1
            if self.actual_steps>=target:
                self.stop=True
        def validate(self):
            return {},0.
        def final_eval(self):
            pass
    model=YOLO(a.weights)
    model.train(trainer=FixedStepTrainer,data=str(data),project=str(a.work/'training'),name='yolo',
                imgsz=cfg['imgsz'],batch=batch,nbs=batch,epochs=math.ceil(target/math.ceil(k/batch)),
                patience=0,val=False,optimizer='AdamW',lr0=cfg['lr0'],lrf=cfg['final_lr_fraction'],
                warmup_epochs=0.,weight_decay=cfg['weight_decay'],workers=0,device=0,seed=a.seed,
                deterministic=True,amp=True,plots=False,save=True,save_period=-1,exist_ok=False,
                cache=False,overlap_mask=False,mask_ratio=2,**cfg['augmentation'])
    if model.trainer.actual_steps!=target:
        raise RuntimeError('Optimizer-hook budget mismatch.')
    model=YOLO(a.work/'training/yolo/weights/last.pt')
    for row in frames:
        result=model.predict(str(row['image']),imgsz=cfg['imgsz'],conf=cfg['conf'],retina_masks=cfg['retina_masks'],device=0,verbose=False)[0]
        semantic=np.zeros(result.orig_shape,np.uint8)
        if result.masks is not None:
            scores=result.boxes.conf.cpu().numpy()
            for i in np.argsort(scores,kind='stable'):
                semantic[result.masks.data[int(i)].cpu().numpy()>0]=int(result.boxes.cls[int(i)])+1
        save_mask(a.output/row['sequence']/(Path(row['frame_id']).stem+'.png'),semantic)


if __name__=='__main__':
    main()
