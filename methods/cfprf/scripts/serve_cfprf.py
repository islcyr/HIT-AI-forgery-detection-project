"""Resident CFPRF process; only listens on loopback, separate from FastAPI."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
import traceback

METHOD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT))
import soundfile as sf
from audio_forgery import CFPRFDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for item in config['artifacts']:
        digest = hashlib.sha256()
        with (PROJECT_ROOT / item['path']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            raise RuntimeError(f"Artifact changed: {item['path']}; rebuild the model configuration")
    model = CFPRFDetector(variant=config['checkpoint_variant'], device=args.device)
    lock = threading.Lock()
    stats = {'model_load_count':1, 'requests_completed':0, 'pid':os.getpid()}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error':'not_found'})
            self.reply(200, {'ready':True, 'descriptor':config['descriptor'], **stats})

        def do_POST(self):
            if self.path != '/predict':
                return self.reply(404, {'error':'not_found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self.reply(400, {'error':'invalid_length'})
            if not 0 < length <= 2 * 1024 * 1024:
                return self.reply(413, {'error':'invalid_audio_size'})
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                return self.reply(400, {'error':'incomplete_body'})
            if not lock.acquire(timeout=5):
                return self.reply(503, {'error':'model_busy'})
            try:
                wave, rate = sf.read(io.BytesIO(body), dtype='float32')
                result = model.predict(wave, rate)
                stats['requests_completed'] += 1
                self.reply(200, {'result':result, **stats})
            except ValueError as error:
                self.reply(422, {'error':str(error)})
            except Exception:
                traceback.print_exc()
                self.reply(500, {'error':'inference_failed'})
            finally:
                lock.release()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    print(f'CFPRF loaded once; PID={os.getpid()}, device={model.device}, port={args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
