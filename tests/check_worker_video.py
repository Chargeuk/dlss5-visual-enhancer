"""Exercise isolated file-video and Live pipelines with a short synthetic clip."""
import sys, subprocess, tempfile, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core.paths import FFMPEG
from src.core.jobs import JobController
from src.core.neural_worker import WORKER
from src.core.neural_bridge import BRIDGE_MANAGER
from src.neural_rendering.video.processor import convert_video
from src.neural_rendering.video.models import ConversionOptions
from src.live.pipeline import LiveSession
from src.live.models import LiveOptions

with tempfile.TemporaryDirectory(prefix='worker-video-check-') as directory:
    root=Path(directory); source=root/'input.mp4'
    subprocess.run([str(FFMPEG),'-hide_banner','-loglevel','error','-f','lavfi','-i',
        'testsrc2=size=512x288:rate=12:duration=1','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
    for gpu_mode in (True,False):
        result=convert_video(source,ConversionOptions(nr_gpu_mode=gpu_mode),output_dir=root/str(gpu_mode),controller=JobController())
        assert result.frames==12,result
        assert Path(result.output_path).is_file()
        assert BRIDGE_MANAGER._initialized_ordinal is None
        print('PASS isolated file video, VRAM='+str(gpu_mode),flush=True)
    # Live uses a separate owned worker so playback need not hold the render queue.
    live=LiveSession(LiveOptions(source=str(source),max_height=480,target_fps='Source',
        open_mpv=False,segment_seconds=1,buffer_seconds=2))
    live.start()
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            info=live.snapshot()
            if not info.processing: break
            time.sleep(.2)
        assert not info.processing and info.processed_frames==12 and not info.failures,info
        assert BRIDGE_MANAGER._initialized_ordinal is None
        print('PASS isolated Live rendering and parent status updates',flush=True)
    finally:
        live.stop(); live.join(timeout=15)
        assert not live.is_alive()
WORKER.stop()
