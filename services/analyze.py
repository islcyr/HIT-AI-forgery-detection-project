"""Upload one audio and wait for the actual VoxGuard result."""
import argparse
import json
from pathlib import Path
import sys
import time
import httpx


def main():
    parser = argparse.ArgumentParser(description='调用五模型综合检测，上传一条音频并取得结果')
    parser.add_argument('audio',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--model',default='ensemble-five')
    parser.add_argument('--mode',choices=['detect','detect_and_localize'],default='detect_and_localize')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    try:
        with httpx.Client(base_url=args.url,timeout=30,trust_env=False) as client:
            with args.audio.open('rb') as audio:
                response = client.post('/api/v1/analyses',files={'file':(args.audio.name,audio)},
                                       data={'model_id':args.model,'mode':args.mode,'include_timeline':'false'})
            response.raise_for_status()
            record = response.json()
            print(f"analysis_id={record['analysis_id']}",file=sys.stderr)
            deadline = time.monotonic() + 620
            while record['status'] not in ('succeeded','failed'):
                if time.monotonic() > deadline:
                    raise TimeoutError('等待超时，服务端任务状态未被取消')
                time.sleep(0.2)
                response = client.get('/api/v1/analyses/'+record['analysis_id'])
                response.raise_for_status()
                record = response.json()
            payload = json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(payload,encoding='utf-8')
            else:
                print(payload,end='')
            return 0 if record['status'] == 'succeeded' else 1
    except (OSError,ValueError,TimeoutError,httpx.HTTPError) as error:
        print(f'调用失败：{error}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
