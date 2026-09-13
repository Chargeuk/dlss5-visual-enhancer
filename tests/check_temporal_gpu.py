"""Real GPU checks for ordered enhancement; no server image files are written."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.core.jobs import JobController, active_job
from src.neural_rendering import sequence as module

y,x=np.indices((96,128))
base=np.stack((20+x%64,20+y%64,20+(x+y)%64,np.full_like(x,180)),axis=-1).astype(np.uint8)
frames=[np.roll(base,i*2,axis=1) for i in range(3)]+[np.full_like(base,235)]
frames[-1][...,3]=180
encoded=[]
for pixels in frames:
    with Image.fromarray(pixels) as image,io.BytesIO() as buffer:
        image.save(buffer,format='PNG'); encoded.append(buffer.getvalue())
save=Image.Image.save
process=module.DLSSFrameSession.process

def memory_save(image,target,*args,**kwargs):
    assert hasattr(target,'write'), f'Unexpected image file write: {target}'
    return save(image,target,*args,**kwargs)

for label,params in [
    ('neural',dict(target_width=128,target_height=96,parameters={'nr_passes':2,'shimmer_suppression':.7})),
    ('vsr-then-neural',dict(target_width=192,target_height=144,parameters={'nr_passes':2,'shimmer_suppression':.7})),
    ('vsr-only',dict(target_width=192,target_height=144,enable_neural_rendering=False)),
]:
    setup=module.validate_setup(dict(version=1,width=128,height=96,frame_count=4,channels=4,**params))
    resets=[]
    def observe(session,**kwargs):
        resets.append(kwargs['reset'])
        return process(session,**kwargs)
    controller=JobController()
    with active_job(controller),patch.object(Image.Image,'save',memory_save),patch.object(module.DLSSFrameSession,'process',observe):
        sequence=module.EnhancementSequence(setup,controller)
        try:
            sequence.open()
            for index,payload in enumerate(encoded):
                result,cut=sequence.push(payload,index)
                with Image.open(io.BytesIO(result)) as image:
                    assert image.size==(params['target_width'],params['target_height'])
                    assert image.mode=='RGBA'
                    assert np.all(np.asarray(image)[...,3]==180)
            sequence.close()
        except BaseException:
            sequence.close(abort=True)
            raise
    if setup.get('enable_neural_rendering',True):
        assert resets==[True,False,False,True],resets
        assert sequence.stats['neural_evaluations']==8,sequence.stats
        assert sequence.stats['scene_cuts']==1,sequence.stats
        assert sequence.stats['temporal']['stabilized_frames']==2,sequence.stats
    assert sequence.stats['vsr_frames']==(0 if label=='neural' else 4)
    print('PASS',label,json.dumps(sequence.stats),flush=True)
print('ALL TEMPORAL GPU CHECKS PASSED',flush=True)
