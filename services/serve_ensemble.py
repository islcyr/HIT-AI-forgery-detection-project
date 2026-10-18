"""Loopback HTTP coordinator for five persistent, isolated model workers."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import traceback

from model_router import MODEL_SPECS, fuse_results

ROOT = Path(__file__).resolve().parents[1]


def read_reply(process, timeout):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise RuntimeError('Model worker response timed out')
    line = process.stdout.readline()
    if not line:
        raise RuntimeError('Model worker exited')
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workers = {}
    stats = {}
    lock = threading.Lock()
    server = None
    def terminate(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        for artifact in config['artifacts']:
            digest = hashlib.sha256()
            with (ROOT / artifact['path']).open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != artifact['sha256']:
                raise RuntimeError(f"Artifact changed: {artifact['path']}")
        for model_id, method, _, _ in MODEL_SPECS:
            process = subprocess.Popen([str(ROOT / '.envs' / method / 'bin/python'),
                                        str(ROOT / 'services/ensemble_worker.py'), model_id, args.device],
                                       cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, bufsize=1)
            workers[model_id] = process
            ready = read_reply(process, 180)
            if ready.get('model_id') != model_id or not ready.get('ready') or ready.get('pid') != process.pid:
                raise RuntimeError(f'{model_id}: invalid startup response')
            stats[model_id] = {**ready, 'requests_completed': 0}
            print(f'{model_id} loaded once; PID={process.pid}', flush=True)
        completed = 0
        failed = False

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, data):
                body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path != '/health':
                    return self.reply(404, {'error': 'not_found'})
                ready = not failed and all(p.poll() is None for p in workers.values())
                self.reply(200, {'ready': ready, 'descriptor': config['descriptor'], 'pid': os.getpid(),
                                 'model_load_count': 5, 'requests_completed': completed, 'workers': stats})

            def do_POST(self):
                nonlocal completed, failed
                if self.path != '/predict':
                    return self.reply(404, {'error': 'not_found'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    return self.reply(400, {'error': 'invalid_length'})
                if not 0 < length <= 2 * 1024 * 1024:
                    return self.reply(413, {'error': 'invalid_audio_size'})
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    return self.reply(400, {'error': 'incomplete_body'})
                if not lock.acquire(timeout=5):
                    return self.reply(503, {'error': 'model_busy'})
                try:
                    if failed or any(p.poll() is not None for p in workers.values()):
                        return self.reply(503, {'error': 'model_unavailable'})
                    request = json.dumps({'audio': base64.b64encode(body).decode()}) + '\n'
                    results = {}
                    for model_id, process in workers.items():
                        process.stdin.write(request)
                        process.stdin.flush()
                        response = read_reply(process, 120)
                        if 'error' in response:
                            return self.reply(422, {'error': 'invalid_audio_or_worker_output', 'model_id': model_id})
                        results[model_id] = response['result']
                        stats[model_id]['requests_completed'] += 1
                    result = fuse_results(results)
                    completed += 1
                    self.reply(200, {'result': result, 'pid': os.getpid(), 'requests_completed': completed})
                except Exception:
                    # A timed-out worker may still have a pending reply. Refuse
                    # further tasks rather than cross-wire replies or reload it.
                    failed = True
                    traceback.print_exc()
                    self.reply(503, {'error': 'ensemble_unavailable'})
                finally:
                    lock.release()

        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        server.daemon_threads = True
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        for process in workers.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
