"""Opt-in native GPU check for the combined VTS temporal pipeline."""
import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.core import jobs
from src.core.neural_worker import WORKER
from src.neural_rendering.sequence import CombinedEnhancementSequence, validate_setup


def main():
    x,y=np.meshgrid(np.arange(128),np.arange(96))
    rgb=np.stack(((x*2)%256,(y*2)%256,((x+y)*2)%256),axis=-1).astype(np.uint8)
    for neural,multiplier,target in ((True,2,(192,144)),(False,3,(128,96)),(False,8,(128,96))):
        setup=validate_setup(dict(version=2,width=128,height=96,target_width=target[0],target_height=target[1],
            channels=4,frame_count=3,enable_neural_rendering=neural,enable_frame_interpolation=True,
            interpolation_multiplier=multiplier,parameters={'nr_passes':1,'shimmer_suppression':.7} if neural else {}))
        with jobs.active_job() as controller:
            stream=CombinedEnhancementSequence(setup,controller)
            output_count=0
            complete=False
            try:
                stream.open()
                for index in range(3):
                    rgba=np.concatenate((np.roll(rgb,index,axis=1),np.full((96,128,1),180,np.uint8)),axis=2)
                    with Image.fromarray(rgba) as image,io.BytesIO() as buffer:
                        image.save(buffer,format='PNG',compress_level=1)
                        frames,cut=stream.push_frames(buffer.getvalue(),index)
                    assert len(frames)==(1 if index==0 else multiplier)
                    for frame in frames:
                        assert frame.shape==(target[1],target[0],4)
                        assert np.all(frame[...,3]==180)
                    output_count+=len(frames)
                complete=True
            finally: stream.close(abort=not complete)
        assert output_count==2*multiplier+1
        assert stream.stats['output_frames']==output_count
        if neural:
            assert stream.stats['neural_evaluations']==3
            assert stream.stats['vsr_frames']==3
        print(f'PASS native combined stream: neural={neural}, target={target}, multiplier={multiplier}, outputs={output_count}',flush=True)


if __name__=='__main__':
    try: main()
    finally:
        jobs._cancel_idle_timer()
        WORKER.stop()
