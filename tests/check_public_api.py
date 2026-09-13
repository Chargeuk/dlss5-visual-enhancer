import base64, io, json, sys
from pathlib import Path
from PIL import Image
import numpy as np
from gradio_client import Client
from websockets.sync.client import connect

url=sys.argv[1].rstrip('/') if len(sys.argv)>1 else 'http://127.0.0.1:7866'
with Image.new('RGBA',(128,96),(150,100,50,180)) as source, io.BytesIO() as buffer:
    source.save(buffer,format='PNG'); encoded=base64.b64encode(buffer.getvalue()).decode()
client=Client(url,download_files=False,verbose=False)
for name,params,size in [('modern',{'iterations':2,'nr_passes':2},(128,96)),('vsr-then-neuroframe',{'iterations':3,'nr_passes':2,'target_width':192,'target_height':144},(192,144)),('vsr',{'operation':'vsr','target_width':192,'target_height':144},(192,144))]:
    returned=client.predict(encoded,json.dumps(params),'public-'+name,api_name='/vts_enhance_memory')
    with Image.open(io.BytesIO(base64.b64decode(returned))) as image: assert image.size==size
    print('PASS public image API',name,flush=True)
assert client.predict('absent-test',api_name='/vts_cancel') is False
print('PASS public cancellation API',flush=True)

with connect(('wss://' if url.startswith('https:') else 'ws://') + url.split('://',1)[1] + '/vts/interpolate',proxy=None,compression=None) as socket:
    socket.send(json.dumps(dict(version=1,width=512,height=288,frame_count=3,multiplier=3)))
    ready=json.loads(socket.recv(timeout=60)); assert ready['type']=='ready',ready
    assert ready['output_count']==7
    generated=0
    y,x=np.indices((288,512))
    rgba=np.stack((x%128+64,y%128+64,(x+y)%128+64,np.full_like(x,255)),axis=-1).astype(np.uint8)
    for index in range(3):
        with Image.fromarray(np.roll(rgba,index*2,axis=1)) as image,io.BytesIO() as buffer:
            image.save(buffer,format='PNG'); data=buffer.getvalue()
        socket.send(json.dumps(dict(type='frame',index=index,bytes=len(data))))
        for offset in range(0,len(data),262144): socket.send(data[offset:offset+262144])
        while True:
            response=json.loads(socket.recv(timeout=60))
            if response['type']=='frame_done': break
            assert response['type']=='generated',response
            payload=bytearray()
            while len(payload)<response['bytes']: payload.extend(socket.recv(timeout=60))
            with Image.open(io.BytesIO(payload)) as image: assert image.size==(512,288)
            generated+=1
    socket.send(json.dumps(dict(type='end')))
    assert json.loads(socket.recv(timeout=30))['type']=='done'
    assert generated==4
print('PASS public interpolation stream: 3 uploads, 4 generated downloads, 7 output slots',flush=True)
