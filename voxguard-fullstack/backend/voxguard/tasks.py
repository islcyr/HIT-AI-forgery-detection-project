import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import uuid

from .schemas import Descriptor
from .settings import Settings


def utc_now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RequestError(Exception):
    def __init__(self, status, code, message, retryable=False):
        self.status = status
        self.detail = {"code": code, "message": message, "retryable": retryable}


class TaskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.records = {}
        self.models = {}
        self.queue = asyncio.Queue()
        self.pending = 0
        self.lock = asyncio.Lock()
        self.process = None
        self.runner = None
        self.cleaner = None
        self.file_lock = None

    async def start(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH")
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_lock = (self.settings.data_dir / ".service.lock").open("a+")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                self.file_lock.seek(0)
                self.file_lock.write("0")
                self.file_lock.flush()
                self.file_lock.seek(0)
                msvcrt.locking(self.file_lock.fileno(), msvcrt.LK_NBLCK, 1)
            # Discover in a disposable process so loading GPU weights at startup
            # does not leave a CUDA context or allocator cache in the HTTP process.
            config_path = self.settings.data_dir / ".models-request.json"
            output_path = self.settings.data_dir / ".models.json"
            write_json(config_path, {"enable_mock": self.settings.enable_mock, "adapter_factory": self.settings.adapter_factory})
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
            with (self.settings.data_dir / "startup.log").open("wb") as log:
                discovery = await asyncio.create_subprocess_exec(sys.executable, "-m", "voxguard.worker", "--describe",
                             str(config_path), str(output_path), env=env, stdout=log, stderr=log,
                             start_new_session=os.name == "posix")
                try:
                    await asyncio.wait_for(discovery.wait(), self.settings.model_load_timeout_sec)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    await self.stop_process(discovery)
                    raise RuntimeError("Model startup timed out; inspect data/startup.log")
            if discovery.returncode != 0 or not output_path.is_file():
                raise RuntimeError("Model startup failed; inspect data/startup.log")
            descriptors = json.loads(output_path.read_text(encoding="utf-8"))
            self.models = {item["id"]: Descriptor.model_validate(item).model_dump() for item in descriptors}
            for orphan in self.settings.data_dir.glob("upload-*.tmp"):
                orphan.unlink(missing_ok=True)
            for directory in self.settings.data_dir.iterdir():
                path = directory / "record.json"
                if not directory.is_dir() or not path.is_file():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    self.records[directory.name] = record
                    if record.get("status") in ("queued", "running"):
                        self.fail(directory.name, "SERVICE_RESTARTED", "服务重启中断了任务，请重新提交。", True)
                except (OSError, ValueError, KeyError):
                    logging.exception("Could not recover task %s", directory.name)
            self.expire()
            self.runner = asyncio.create_task(self.run())
            self.cleaner = asyncio.create_task(self.cleanup_loop())
        except Exception:
            self.file_lock.close()
            self.file_lock = None
            raise

    async def close(self):
        for task in (self.runner, self.cleaner):
            if task:
                task.cancel()
        for task in (self.runner, self.cleaner):
            if task:
                with suppress(asyncio.CancelledError):
                    await task
        for analysis_id, record in list(self.records.items()):
            if record.get("status") in ("queued", "running"):
                self.fail(analysis_id, "SERVICE_RESTARTED", "服务停止中断了任务，请重新提交。", True)
        if self.file_lock:
            self.file_lock.close()
            self.file_lock = None

    def save(self, analysis_id):
        write_json(self.settings.data_dir / analysis_id / "record.json", self.records[analysis_id])

    def fail(self, analysis_id, code, message, retryable=False):
        record = self.records[analysis_id]
        record.update(status="failed", completed_at=iso(utc_now()), result=None,
                      error={"code": code, "message": message, "retryable": retryable})
        self.save(analysis_id)

    async def accept(self, upload_path: Path, model_id: str, mode: str, include_timeline: bool):
        async with self.lock:
            if self.pending >= self.settings.queue_capacity:
                raise RequestError(429, "QUEUE_FULL", "分析队列已满，请稍后重试。", True)
            descriptor = self.models[model_id]
            analysis_id = ("example-" if descriptor["is_mock"] else "analysis-") + uuid.uuid4().hex
            directory = self.settings.data_dir / analysis_id
            directory.mkdir()
            try:
                upload_path.replace(directory / "upload.bin")
                now = utc_now()
                record = {"analysis_id": analysis_id, "status": "queued", "mode": mode,
                          "created_at": iso(now), "completed_at": None,
                          "expires_at": iso(now + timedelta(seconds=self.settings.retention_sec)),
                          "model": {"id": descriptor["id"], "version": descriptor["version"]},
                          "is_mock": descriptor["is_mock"], "audio": None, "result": None, "error": None,
                          "warnings": [{"code": "MOCK_DATA", "message": "模拟输出用于接口联调，不代表音频真假或真实模型能力。"}] if descriptor["is_mock"] else []}
                write_json(directory / "request.json", {"analysis_id": analysis_id, "descriptor": descriptor,
                           "mode": mode, "include_timeline": include_timeline, "limits": self.settings.limits(),
                           "enable_mock": self.settings.enable_mock, "adapter_factory": self.settings.adapter_factory})
                self.records[analysis_id] = record
                self.save(analysis_id)
                self.pending += 1
                self.queue.put_nowait(analysis_id)
                # Return a snapshot: the worker may immediately advance the stored state.
                return dict(record)
            except Exception:
                self.records.pop(analysis_id, None)
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def expire(self):
        now = utc_now()
        for analysis_id, record in list(self.records.items()):
            if record.get("expired") or parse_time(record["expires_at"]) > now:
                continue
            if record.get("status") in ("queued", "running"):
                continue
            directory = self.settings.data_dir / analysis_id
            self.records[analysis_id] = {"analysis_id": analysis_id, "expired": True}
            # Keep only the minimal tombstone needed to distinguish 410 from 404.
            self.save(analysis_id)
            for path in directory.iterdir():
                if path.name != "record.json" and path.is_file():
                    path.unlink(missing_ok=True)

    def get(self, analysis_id):
        self.expire()
        record = self.records.get(analysis_id)
        if record is None:
            raise RequestError(404, "ANALYSIS_NOT_FOUND", "分析任务不存在。")
        if record.get("expired"):
            raise RequestError(410, "ANALYSIS_EXPIRED", "分析任务及音频已过保留期限。")
        return record

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(30)
            self.expire()

    async def stop_process(self, process):
        if process.returncode is None:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
        await process.wait()

    async def run(self):
        while True:
            analysis_id = await self.queue.get()
            directory = self.settings.data_dir / analysis_id
            process = None
            try:
                record = self.records[analysis_id]
                remaining = self.settings.task_timeout_sec - (utc_now() - parse_time(record["created_at"])).total_seconds()
                if remaining <= 0:
                    self.fail(analysis_id, "INFERENCE_TIMEOUT", "任务等待时间超过处理期限。", True)
                    continue
                record["status"] = "running"
                self.save(analysis_id)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
                with (directory / "worker.log").open("wb") as log:
                    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "voxguard.worker", str(directory),
                              env=env, stdout=log, stderr=log, start_new_session=os.name == "posix")
                    self.process = process
                    try:
                        await asyncio.wait_for(process.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        await self.stop_process(process)
                        self.fail(analysis_id, "INFERENCE_TIMEOUT", "模型处理超过任务时间限制。", True)
                        continue
                output_path = directory / "output.json"
                if process.returncode != 0 or not output_path.is_file():
                    self.fail(analysis_id, "INFERENCE_FAILED", "模型进程异常退出，请查看服务日志。")
                    continue
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                if "error" in payload:
                    error = payload["error"]
                    self.fail(analysis_id, error["code"], error["message"], error["retryable"])
                else:
                    record.update(status="succeeded", completed_at=iso(utc_now()), audio=payload["audio"], result=payload["result"])
                    record["warnings"].extend(payload["warnings"])
                    self.save(analysis_id)
            except asyncio.CancelledError:
                if process:
                    await self.stop_process(process)
                raise
            except Exception:
                logging.exception("Task processing failed: %s", analysis_id)
                self.fail(analysis_id, "INTERNAL_ERROR", "任务处理异常，请查看服务日志。")
            finally:
                self.process = None
                self.pending -= 1
                self.queue.task_done()
