import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.core.jobs import JobController, active_job
from src.frame_interpolation.stream import InterpolationStream

y, x = np.indices((288, 512))
rgb = np.stack(((x % 128) + 64, (y % 128) + 64, ((x+y) % 128) + 64,
                np.full_like(x, 180)), axis=-1).astype(np.uint8)
def png(pixels):
    with Image.fromarray(pixels) as image, io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        return buffer.getvalue()

for multiplier in (2, 3, 4, 8):
    print('RUN', multiplier, flush=True)
    controller = JobController()
    with active_job(controller):
        stream = InterpolationStream(dict(width=512, height=288, frame_count=3, multiplier=multiplier), controller)
        try:
            stream.open()
            for index in range(3):
                outputs, cut = stream.push(png(np.roll(rgb, 2 * index, axis=1)), index)
                assert not cut
                assert len(outputs) == (multiplier - 1 if index else 0)
                for header, data in outputs:
                    assert header['type'] == 'generated'
                    with Image.open(io.BytesIO(data)) as result:
                        assert result.size == (512, 288)
                        assert result.getpixel((0, 0))[3] == 180
            print('PASS', multiplier, 'positions', stream.positions, flush=True)
        finally:
            stream.close()
print('ALL NATIVE MULTIPLIERS PASSED', flush=True)
