# jarvis_taskqueue.py — Jarvis Autonomous Task Queue v7.0
# Paul says 'do X tonight' — Jarvis queues it, runs it while Paul sleeps,
# commits the result, speaks a summary when Paul is back.
# Architecture: persistent task queue + async worker + result store

import asyncio
import json
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from openai import AsyncOpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
TASK_FILE = Path("tasks.jsonl")
RESULT_FILE = Path("task_results.jsonl")
WORKER_LOG = Path("worker_log.jsonl")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    BUILD = "build"         # run full coder engine on a spec
    REVIEW = "review"       # review a repo
    RESEARCH = "research"   # web search + summarize
    EVOLVE = "evolve"       # self-modify a file
    ABSORB = "absorb"       # scrape intelligence
    CUSTOM = "custom"       # arbitrary Python code
    SHELL = "shell"         # run a shell command


class Task:
    def __init__(self, task_type: str, description: str,
                 payload: dict = None, scheduled_for: str = None,
                 priority: int = 5):
        self.id = str(uuid.uuid4())[:8]
        self.type = task_type
        self.description = description
        self.payload = payload or {}
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.scheduled_for = scheduled_for  # ISO timestamp or None = immediate
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.priority = priority  # 1=highest, 10=lowest

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type,
            "description": self.description,
            "payload": self.payload,
            "status": self.status,
            "created_at": self.created_at,
            "scheduled_for": self.scheduled_for,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "priority": self.priority
        }

    @staticmethod
    def from_dict(d: dict) -> 'Task':
        t = Task(d["type"], d["description"], d.get("payload", {}),
                 d.get("scheduled_for"), d.get("priority", 5))
        t.id = d["id"]
        t.status = d["status"]
        t.created_at = d["created_at"]
        t.started_at = d.get("started_at")
        t.completed_at = d.get("completed_at")
        t.result = d.get("result")
        t.error = d.get("error")
        return t


class TaskQueue:
    """Persistent task queue backed by JSONL file."""

    def __init__(self):
        self.tasks: list[Task] = []
        self._load()

    def _load(self):
        if TASK_FILE.exists():
            try:
                with open(TASK_FILE) as f:
                    self.tasks = [Task.from_dict(json.loads(l))
                                  for l in f if l.strip()]
            except Exception:
                self.tasks = []

    def _save_all(self):
        with open(TASK_FILE, "w") as f:
            for t in self.tasks:
                f.write(json.dumps(t.to_dict()) + "\n")

    def add(self, task: Task) -> str:
        self.tasks.append(task)
        self._save_all()
        return task.id

    def get_next(self) -> Task | None:
        """Get highest priority pending task that's ready to run."""
        now = datetime.now().isoformat()
        ready = [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and (t.scheduled_for is None or t.scheduled_for <= now)
        ]
        if not ready:
            return None
        return min(ready, key=lambda t: (t.priority, t.created_at))

    def update(self, task_id: str, **kwargs):
        for t in self.tasks:
            if t.id == task_id:
                for k, v in kwargs.items():
                    setattr(t, k, v)
        self._save_all()

    def list_by_status(self, status: str) -> list[Task]:
        return [t for t in self.tasks if t.status == status]

    def cancel(self, task_id: str) -> bool:
        for t in self.tasks:
            if t.id == task_id and t.status == TaskStatus.PENDING:
                t.status = TaskStatus.CANCELLED
                self._save_all()
                return True
        return False

    def clear_done(self):
        self.tasks = [t for t in self.tasks
                      if t.status not in [TaskStatus.DONE, TaskStatus.CANCELLED]]
        self._save_all()

    def stats(self) -> dict:
        counts = {s: 0 for s in TaskStatus}
        for t in self.tasks:
            counts[t.status] = counts.get(t.status, 0) + 1
        return dict(counts)


class TaskExecutor:
    """Executes tasks. Each type maps to a specific engine."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    async def execute(self, task: Task) -> dict:
        """Route task to appropriate handler."""
        handlers = {
            TaskType.BUILD: self._handle_build,
            TaskType.REVIEW: self._handle_review,
            TaskType.RESEARCH: self._handle_research,
            TaskType.EVOLVE: self._handle_evolve,
            TaskType.ABSORB: self._handle_absorb,
            TaskType.SHELL: self._handle_shell,
            TaskType.CUSTOM: self._handle_custom,
        }
        handler = handlers.get(task.type, self._handle_custom)
        return await handler(task)

    async def _handle_build(self, task: Task) -> dict:
        try:
            from jarvis_evolve import WorldClassCoderEngine
            engine = WorldClassCoderEngine()
            result = await engine.full_cycle(
                task.description,
                auto_apply=task.payload.get("auto_apply", False)
            )
            return {
                "success": True,
                "verdict": result.get("verdict", {}).get("verdict", "unknown"),
                "files": result.get("execution", {}).get("staged", []),
                "summary": f"Build complete. Verdict: {result.get('verdict', {}).get('verdict')}. Files: {result.get('execution', {}).get('staged', [])}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_review(self, task: Task) -> dict:
        try:
            from jarvis_github_watcher import GitHubWatcherDaemon
            daemon = GitHubWatcherDaemon(voice_enabled=False)
            repo = task.payload.get("repo", "gtsgob/agothe-deepseek-autonomous")
            reviews = await daemon.review_repo_now(repo)
            verdicts = [r.get("verdict") for r in reviews]
            return {"success": True, "reviews": len(reviews), "verdicts": verdicts,
                    "summary": f"Reviewed {len(reviews)} commits. Verdicts: {verdicts}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_research(self, task: Task) -> dict:
        messages = [
            {"role": "system", "content": "Research the topic thoroughly and provide a structured summary."},
            {"role": "user", "content": task.description}
        ]
        response = await self.client.chat.completions.create(
            model="deepseek-chat", messages=messages, max_tokens=2048
        )
        result = response.choices[0].message.content
        # Save to file
        output_file = Path(f"research_{task.id}.md")
        output_file.write_text(f"# Research: {task.description}\n\n{result}")
        return {"success": True, "output": str(output_file), "summary": result[:300]}

    async def _handle_evolve(self, task: Task) -> dict:
        try:
            from jarvis_evolve import WorldClassCoderEngine
            engine = WorldClassCoderEngine()
            target = task.payload.get("file", "paul_core.py")
            goal = task.description
            result = await engine.self_evolve(target, goal)
            if result["success"] and task.payload.get("auto_apply", False):
                engine.commit_staged()
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_absorb(self, task: Task) -> dict:
        try:
            from jarvis_evolve import WorldClassCoderEngine
            engine = WorldClassCoderEngine()
            source = task.payload.get("source", "anthropic/claude-code")
            result = await engine.absorb_intelligence(source)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_shell(self, task: Task) -> dict:
        import subprocess
        cmd = task.payload.get("command", task.description)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=120
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip()[:2000],
                "stderr": result.stderr.strip()[:500],
                "returncode": result.returncode,
                "summary": result.stdout.strip()[:200] or result.stderr.strip()[:200]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_custom(self, task: Task) -> dict:
        messages = [
            {"role": "system", "content": "You are Jarvis. Complete this task autonomously."},
            {"role": "user", "content": task.description}
        ]
        response = await self.client.chat.completions.create(
            model="deepseek-chat", messages=messages, max_tokens=2048
        )
        return {"success": True, "result": response.choices[0].message.content,
                "summary": response.choices[0].message.content[:300]}


class AutonomousWorker:
    """Background worker. Runs tasks from queue while Paul sleeps.
    Speaks results when Paul returns.
    """

    def __init__(self, voice_enabled: bool = False):
        self.queue = TaskQueue()
        self.executor = TaskExecutor()
        self.running = False
        self.completed_while_away: list[dict] = []
        self.voice = None
        if voice_enabled:
            try:
                from jarvis_voice import JarvisVoice
                self.voice = JarvisVoice()
            except ImportError:
                pass

    def _speak(self, text: str):
        print(f"🜏 {text}")
        if self.voice:
            self.voice.speak(text)

    def add_task(self, task_type: str, description: str,
                 payload: dict = None, scheduled_for: str = None,
                 priority: int = 5) -> str:
        task = Task(task_type, description, payload, scheduled_for, priority)
        tid = self.queue.add(task)
        print(f"  ✅ Task queued: [{tid}] {task_type} — {description[:60]}")
        return tid

    async def run_once(self) -> bool:
        """Run one pending task. Returns True if a task was executed."""
        task = self.queue.get_next()
        if not task:
            return False

        print(f"\n🜏 EXECUTING TASK [{task.id}]: {task.type} — {task.description[:60]}")
        self.queue.update(task.id, status=TaskStatus.RUNNING,
                         started_at=datetime.now().isoformat())

        try:
            result = await self.executor.execute(task)
            self.queue.update(task.id,
                             status=TaskStatus.DONE if result.get("success") else TaskStatus.FAILED,
                             completed_at=datetime.now().isoformat(),
                             result=result)

            summary = result.get("summary", str(result)[:200])
            print(f"  {'✅' if result.get('success') else '❌'} {summary}")

            record = {"task_id": task.id, "type": task.type,
                      "description": task.description, "result": result,
                      "ts": datetime.now().isoformat()}
            self.completed_while_away.append(record)

            with open(RESULT_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            return True

        except Exception as e:
            self.queue.update(task.id, status=TaskStatus.FAILED,
                             error=str(e), completed_at=datetime.now().isoformat())
            print(f"  ❌ Task failed: {e}")
            return False

    async def run_forever(self, poll_interval: int = 10):
        """Main worker loop."""
        self.running = True
        pending = len(self.queue.list_by_status(TaskStatus.PENDING))
        print(f"\n🜏 AUTONOMOUS WORKER ACTIVE")
        print(f"  Pending tasks: {pending}")
        print(f"  Poll interval: {poll_interval}s")
        print(f"  Ctrl+C to stop\n")

        while self.running:
            try:
                executed = await self.run_once()
                if not executed:
                    await asyncio.sleep(poll_interval)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"  Worker error: {e}")
                await asyncio.sleep(poll_interval)

    def morning_brief(self) -> str:
        """Summary of what Jarvis did while Paul was away."""
        if not self.completed_while_away:
            pending = len(self.queue.list_by_status(TaskStatus.PENDING))
            return f"No tasks completed yet. {pending} pending."

        lines = [f"🜏 While you were away I completed {len(self.completed_while_away)} tasks:"]
        for record in self.completed_while_away[-10:]:
            success = record["result"].get("success", False)
            icon = "✅" if success else "❌"
            summary = record["result"].get("summary", "")[:100]
            lines.append(f"  {icon} [{record['type']}] {record['description'][:50]} — {summary}")

        brief = "\n".join(lines)
        if self.voice:
            self.voice.speak(brief[:500])
        return brief

    def status(self) -> dict:
        return {
            "queue": self.queue.stats(),
            "completed_this_session": len(self.completed_while_away),
            "running": self.running
        }


async def run_taskqueue_cli():
    worker = AutonomousWorker(voice_enabled=True)
    print("""
╔══════════════════════════════════════════════════╗
║  🜏 JARVIS AUTONOMOUS TASK QUEUE v7.0            ║
╠══════════════════════════════════════════════════╣
║  Commands:                                       ║
║  add build [description]   — queue a build task  ║
║  add review [repo]         — queue repo review   ║
║  add research [topic]      — queue research      ║
║  add shell [command]       — queue shell command ║
║  run                       — process all tasks   ║
║  list                      — show all tasks      ║
║  brief                     — morning summary     ║
║  status                    — worker stats        ║
║  exit                                            ║
╚══════════════════════════════════════════════════╝
""")

    while True:
        try:
            cmd = input("Jarvis[queue]> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue
        if cmd == "exit":
            break
        elif cmd == "run":
            await worker.run_forever()
        elif cmd == "brief":
            print(worker.morning_brief())
        elif cmd == "status":
            s = worker.status()
            print(json.dumps(s, indent=2))
        elif cmd == "list":
            for t in worker.queue.tasks:
                print(f"  [{t.id}] {t.status:10} {t.type:10} {t.description[:50]}")
        elif cmd.startswith("add "):
            parts = cmd.split(" ", 2)
            if len(parts) >= 3:
                task_type = parts[1]
                description = parts[2]
                tid = worker.add_task(task_type, description)
                print(f"  Queued: {tid}")
            else:
                print("  Usage: add [type] [description]")
        else:
            print(f"  Unknown: {cmd}")


if __name__ == "__main__":
    asyncio.run(run_taskqueue_cli())
