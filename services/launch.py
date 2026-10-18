"""Launch the resident model and the isolated API environment together."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / 'services/.voxguard.pid'


def running_launcher():
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text())
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(Path(__file__).resolve()).encode() in command:
            return pid
    except (OSError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-port', type=int, default=8765)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stop', action='store_true', help='停止本项目已启动的后端和模型')
    args = parser.parse_args()
    current = running_launcher()
    if args.stop:
        if current:
            os.kill(current, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while PID_FILE.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            if PID_FILE.exists():
                raise RuntimeError('Service shutdown did not finish within 15 seconds')
            print('VoxGuard and ensemble stopped', flush=True)
        else:
            print('No managed VoxGuard service is running', flush=True)
        return
    if current:
        raise RuntimeError('Service is already running; use bash start.sh --stop first')
    children = []
    for name in ('cfprf','bam','sal','voxguard'):
        if not (ROOT / f'.envs/{name}/bin/python').is_file():
            raise RuntimeError(f'Missing .envs/{name}; see README')
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    config = ROOT / 'voxguard-fullstack/model_integration/ensemble-resident.json'
    endpoint = f'http://127.0.0.1:{args.model_port}'
    client = build_opener(ProxyHandler({}))
    # Do not attach to a pre-existing process at the requested port.
    import socket
    for port in (args.port, args.model_port):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port))
    PID_FILE.write_text(str(os.getpid()))
    try:
        model = subprocess.Popen([sys.executable, str(ROOT/'services/serve_ensemble.py'),
                                  '--config',str(config),'--port',str(args.model_port),'--device',args.device], cwd=ROOT)
        children.append(model)
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if model.poll() is not None:
                raise RuntimeError('Ensemble startup failed; see model output above')
            try:
                with client.open(endpoint + '/health',timeout=1) as response:
                    health = json.loads(response.read())
                if health.get('ready') and health.get('pid') == model.pid:
                    break
            except OSError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError('Ensemble startup timed out')
        env = os.environ.copy()
        env.update(ENABLE_MOCK='false', VOXGUARD_ADAPTER_FACTORY='model_integration.ensemble_resident:create_adapter',
                   VOXGUARD_MODEL_CONFIG=str(config), VOXGUARD_MODEL_ENDPOINT=endpoint)
        backend = subprocess.Popen([str(ROOT/'.envs/voxguard/bin/python'), str(ROOT/'voxguard-fullstack/run.py'),
                                    '--host',args.host,'--port',str(args.port)], cwd=ROOT/'voxguard-fullstack', env=env)
        children.append(backend)
        print(f'VoxGuard: http://{args.host}:{args.port} ; API: /docs ; model PID={model.pid}',flush=True)
        while True:
            if model.poll() is not None or backend.poll() is not None:
                raise RuntimeError('A service exited; shutting down the other service')
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if PID_FILE.exists() and PID_FILE.read_text() == str(os.getpid()):
            PID_FILE.unlink()


if __name__ == '__main__':
    main()
