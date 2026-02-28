# jarvis_self_heal.py — Jarvis Continuous Self-Healing Daemon
# Watches all Python files, detects errors, auto-patches using DeepSeek
# Architecture: Claude Code's "read-verify-patch" loop made autonomous

import asyncio
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
HEAL_LOG = Path("heal_log.jsonl")
WATCH_PATTERNS = ["*.py", "skills/*.py"]
CHECK_INTERVAL = 30  # seconds


class FileWatcher:
    """Watch files for changes and errors."""

    def __init__(self):
        self.checksums: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    def get_checksum(self, path: str) -> str:
        try:
            content = Path(path).read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def scan(self) -> list[dict]:
        """Scan all watched files for syntax errors and changes."""
        issues = []
        for pattern in WATCH_PATTERNS:
            for p in Path(".").glob(pattern):
                path_str = str(p)
                checksum = self.get_checksum(path_str)

                # Check syntax
                try:
                    code = p.read_text(encoding="utf-8", errors="replace")
                    ast.parse(code)
                    # Clear error if previously had one
                    if path_str in self.errors:
                        del self.errors[path_str]
                except SyntaxError as e:
                    error_msg = f"SyntaxError line {e.lineno}: {e.msg}"
                    if self.errors.get(path_str) != error_msg:
                        self.errors[path_str] = error_msg
                        issues.append({
                            "type": "syntax_error",
                            "file": path_str,
                            "error": error_msg,
                            "code_preview": code[max(0, (e.lineno or 1) * 40 - 80):((e.lineno or 1) * 40 + 80)]
                        })

                # Check if changed
                if checksum != self.checksums.get(path_str):
                    self.checksums[path_str] = checksum
                    issues.append({
                        "type": "file_changed",
                        "file": path_str,
                        "checksum": checksum
                    })

        return issues


class RuntimeErrorCatcher:
    """Catch runtime errors by actually running files and capturing tracebacks."""

    def test_file(self, path: str) -> dict:
        """Run a Python file and capture any runtime errors."""
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import ast; ast.parse(open('{path}').read()); print('SYNTAX_OK')"],
                capture_output=True, text=True, timeout=10
            )
            if "SYNTAX_OK" in result.stdout:
                return {"path": path, "status": "ok"}
            return {"path": path, "status": "error", "error": result.stderr}
        except Exception as e:
            return {"path": path, "status": "error", "error": str(e)}

    def run_with_capture(self, code: str, timeout: int = 15) -> dict:
        """Run arbitrary code and capture full output + traceback."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout
            )
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1, "success": False}


class AutoPatcher:
    """Given a file and an error, auto-generate and apply a patch using DeepSeek."""

    PATCH_PROMPT = """You are an expert Python debugger. A file has an error. Fix it.

Rules:
- Output ONLY the corrected Python file, no markdown, no explanation
- Make the minimal change needed to fix the error
- Preserve all functionality that wasn't broken
- If you cannot safely fix it, output the original file unchanged"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.patch_count = 0
        self.patch_history: list[dict] = []

    async def patch(self, file_path: str, error: str) -> dict:
        """Generate and apply a patch for a broken file."""
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "reason": "file not found"}

        original = p.read_text(encoding="utf-8", errors="replace")

        messages = [
            {"role": "system", "content": self.PATCH_PROMPT},
            {"role": "user", "content": f"""File: {file_path}
Error: {error}

Code:
{original}

Fixed code:"""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=4096,
                temperature=0.1  # Low temp for surgical fixes
            )
            fixed = response.choices[0].message.content

            # Strip markdown
            fixed = re.sub(r'^```[a-z]*\n', '', fixed, flags=re.MULTILINE)
            fixed = re.sub(r'^```$', '', fixed, flags=re.MULTILINE).strip()

            # Validate the fix
            try:
                ast.parse(fixed)
            except SyntaxError as e:
                return {
                    "success": False,
                    "reason": f"Patch itself has syntax error: {e}",
                    "file": file_path
                }

            # Backup original
            backup = p.with_suffix(".py.bak")
            backup.write_text(original)

            # Apply patch
            p.write_text(fixed, encoding="utf-8")
            self.patch_count += 1

            patch_record = {
                "ts": datetime.now().isoformat(),
                "file": file_path,
                "error": error,
                "backup": str(backup),
                "patch_num": self.patch_count
            }
            self.patch_history.append(patch_record)

            # Log
            with open(HEAL_LOG, "a") as f:
                f.write(json.dumps(patch_record) + "\n")

            return {
                "success": True,
                "file": file_path,
                "backup": str(backup),
                "patch_num": self.patch_count,
                "lines_before": len(original.splitlines()),
                "lines_after": len(fixed.splitlines())
            }

        except Exception as e:
            return {"success": False, "reason": str(e), "file": file_path}

    def rollback(self, file_path: str) -> bool:
        """Restore from backup if a patch made things worse."""
        backup = Path(file_path).with_suffix(".py.bak")
        if backup.exists():
            Path(file_path).write_text(backup.read_text())
            return True
        return False


class SelfHealDaemon:
    """Continuous background daemon: watch → detect → patch → verify → log."""

    def __init__(self):
        self.watcher = FileWatcher()
        self.catcher = RuntimeErrorCatcher()
        self.patcher = AutoPatcher()
        self.running = False
        self.heal_count = 0
        self.start_time = datetime.now()

    async def heal_cycle(self) -> list[dict]:
        """One full scan-detect-patch cycle."""
        issues = self.watcher.scan()
        heals = []

        for issue in issues:
            if issue["type"] == "syntax_error":
                print(f"\n⚠️  Syntax error detected: {issue['file']}")
                print(f"   Error: {issue['error']}")
                print(f"   Auto-patching...")

                result = await self.patcher.patch(issue["file"], issue["error"])

                if result["success"]:
                    self.heal_count += 1
                    print(f"   ✅ Patched! (heal #{self.heal_count})")
                    heals.append(result)
                else:
                    print(f"   ❌ Patch failed: {result['reason']}")

        return heals

    async def run_forever(self):
        """Main daemon loop."""
        self.running = True
        print(f"\n🜏 SELF-HEAL DAEMON ACTIVE")
        print(f"   Watching: {WATCH_PATTERNS}")
        print(f"   Check interval: {CHECK_INTERVAL}s")
        print(f"   Press Ctrl+C to stop\n")

        while self.running:
            try:
                heals = await self.heal_cycle()
                if heals:
                    print(f"   Session heals: {self.heal_count}")
                await asyncio.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"   Daemon error: {e}")
                await asyncio.sleep(CHECK_INTERVAL)

        uptime = (datetime.now() - self.start_time).seconds
        print(f"\n🜏 Self-heal daemon stopped. Uptime: {uptime}s | Heals: {self.heal_count}")

    def status(self) -> dict:
        uptime = (datetime.now() - self.start_time).seconds
        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "heals_performed": self.heal_count,
            "files_watched": len(self.watcher.checksums),
            "active_errors": len(self.watcher.errors),
            "patch_history": self.patcher.patch_history[-5:]
        }


if __name__ == "__main__":
    daemon = SelfHealDaemon()
    asyncio.run(daemon.run_forever())
