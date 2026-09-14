"""Private subprocess entry point; never exposed as a network endpoint."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['MERSERK_NEURAL_WORKER'] = '1'

from multiprocessing.shared_memory import SharedMemory
import numpy as np
from src.core.neural_worker import send, receive
from src.core.neural_bridge import NeuralBridgePoisonedError
from src.core.jobs import JobController
from src.core.runtime import DLSSFrameSession


def main():
    # Protect the control pipe from Python/native progress output.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    session = memory = token = live = None
    def snapshot():
        return dict(diagnostics=session.diagnostics,timings=session.process_timings,
                    status=session.structured_status(),logs=session.bridge_logs)
    while True:
        try: request = receive(sys.stdin.buffer)
        except EOFError: return
        try:
            op = request['op']
            if op == 'cleanup':
                if session is not None:
                    raise RuntimeError('Cannot clean memory while a neural session is open.')
                from src.core.memory_cleanup import cleanup_local_memory
                cleanup_local_memory()
                response = dict(cleaned=True)
            elif op == 'live_start':
                from src.live.pipeline import LiveSession, validate_options
                validate_options(request['options'])
                live = LiveSession(request['options'])
                live.start()
                response = dict(info=live.snapshot(),ready=False)
            elif op == 'live_status':
                from src.core.neural_bridge import BRIDGE_MANAGER
                BRIDGE_MANAGER._guard_poison()
                response = dict(info=live.snapshot(),ready=live._ready.is_set(),directory=live._session_dir)
            elif op == 'live_stop':
                live.stop(); live.join(timeout=5)
                response = dict(stopped=not live.is_alive())
            elif op == 'live_effects':
                response = dict(accepted=live.request_effects(request['settings']))
            elif op == 'video':
                from src.neural_rendering.video.processor import convert_video
                converted = convert_video(request['source'], request['options'],
                    lambda value, text: send(output,dict(progress=(value,text))),
                    output_dir=request['output_dir'],controller=JobController())
                response = dict(result=converted)
            elif op == 'open':
                if session is not None: raise RuntimeError('A neural session is already open.')
                memory = SharedMemory(name=request['memory'])
                session = DLSSFrameSession(**request['kwargs'],controller=JobController())
                token = request['token']
                shape = (session.output_height,session.output_width,4)
                rgba = np.ndarray(shape,np.uint8,buffer=memory.buf)
                result = np.ndarray(shape,np.uint8,buffer=memory.buf,offset=rgba.nbytes)
                response = snapshot()
            else:
                if token != request['token'] or session is None:
                    raise NeuralBridgePoisonedError('Worker session was lost.')
                if op == 'process':
                    session.native_settings.update(request['settings'])
                    session.process(index=request['index'],rgba=rgba,reset=request['reset'],
                                    pts=request['pts'],output_buffer=result)
                    response = snapshot()
                elif op == 'mask':
                    session.update_composition_mask(request['selection'],request['feather'])
                    response = snapshot()
                elif op == 'close':
                    session.abort() if request['abort'] else session.close()
                    response = snapshot()
                    session = token = None
                    rgba = result = None
                    memory.close(); memory = None
                else: raise ValueError('Unknown worker operation')
            send(output,response)
            # A persistent command loop otherwise retains the previous request
            # (including masks) and video result until another request arrives.
            request = response = converted = None
        except Exception as exc:
            from src.core.neural_bridge import BRIDGE_MANAGER
            fatal = isinstance(exc, (NeuralBridgePoisonedError, OSError)) or bool(BRIDGE_MANAGER._poisoned_reason)
            if not fatal and op == 'open' and memory is not None:
                memory.close(); memory = None
            send(output,dict(error=str(exc),fatal=fatal))
            if fatal: os._exit(1)


if __name__ == '__main__': main()
