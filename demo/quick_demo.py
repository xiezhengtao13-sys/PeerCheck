"""CPU-only replay of saved peer predictions; no models or downloads."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from peercheck.io import read_mask, overlay
from peercheck.verification import agreement, intersection_labels, route
from peercheck.metrics import evaluate


def quick_demo(data, predictions, reference, output, tau=.85, a_min=.0005, compare_paper=True):
    started = time.perf_counter()
    data,predictions,output = Path(data),Path(predictions),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    ref = pd.read_csv(reference)
    sheet = Image.new('RGB',(4*330,7*280+320),'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for i,row in ref.iterrows():
        relative = Path(row.sequence)/(Path(row.frame_id).stem+'.png')
        s,d = read_mask(predictions/'V3'/relative),read_mask(predictions/'VD'/relative)
        v = intersection_labels(s,d)
        if not np.array_equal(v,read_mask(predictions/'V8'/relative)):
            raise ValueError('Saved intersection differs from equal-label intersection.')
        ci = agreement(s,d,a_min)
        metric = evaluate(read_mask(data/'gt'/relative),v,a_min)
        status = route(ci,tau,bool(row.is_keyframe))
        record = {'frame_id':row.frame_id,'frame_index':row.frame_index,'is_keyframe':row.is_keyframe,
                  'C_i':ci,'status':status,**metric}
        if compare_paper:
            record.update(paper_C_i=row.paper_C_i,paper_V8_iou=row.paper_V8_iou,
                          delta_C_i=ci-row.paper_C_i,delta_V8_iou=metric['foreground_iou']-row.paper_V8_iou)
        records.append(record)
        thumbnail = overlay(data/'images'/row.sequence/row.frame_id,v).resize((330,247),Image.Resampling.LANCZOS)
        x,y = (i%4)*330,(i//4)*280
        sheet.paste(thumbnail,(x,y))
        draw.text((x+5,y+249),f"{row.frame_index} {status} C={ci:.3f} IoU={metric['foreground_iou']:.3f}",fill='black')
    # Pillow drawing keeps the CPU demo fast, including its first execution.
    left,top,right,bottom = 60,7*280+20,1280,7*280+260
    for value in [0,.25,.5,.75,1]:
        y = bottom-value*(bottom-top)
        draw.line((left,y,right,y),fill=(220,220,220))
        draw.text((10,y-5),str(value),fill='black')
    for col,color in [('C_i',(0,90,210)),('foreground_iou',(210,65,30))]:
        points = [(left+i*(right-left)/(len(records)-1),bottom-r[col]*(bottom-top)) for i,r in enumerate(records)]
        draw.line(points,fill=color,width=3)
    y = bottom-tau*(bottom-top)
    draw.line((left,y,right,y),fill=(100,100,100),width=1)
    draw.text((60,bottom+15),'Blue: agreement C_i; orange: intersection vs GT IoU; gray: tau=0.85. Horizontal axis: frame order.',fill='black')
    sheet.save(output/'overview.jpg',quality=90)
    table = pd.DataFrame(records)
    table.to_csv(output/'metrics.csv',index=False)
    summary = {'frames':len(table),'accepted':int((table.status=='accept').sum()),'review':int((table.status=='review').sum()),
               'keyframes':int((table.status=='keyframe').sum()),'wall_seconds':time.perf_counter()-started}
    if compare_paper:
        summary.update(max_abs_C_i_difference=float(table.delta_C_i.abs().max()),max_abs_IoU_difference=float(table.delta_V8_iou.abs().max()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('quick_demo_output'))
    a = p.parse_args()
    print(quick_demo(base/'data',base/'precomputed',base/'paper_reference.csv',a.output))


if __name__=='__main__':
    main()
