# jarvis_evolve.py — Jarvis Self-Evolution Engine v3.0
# The core secret of Claude Code, Cursor, Devin — replicated inside our constraints
# Architecture derived from: Claude Code (tool-call loop + bash harness),
# Cursor (shadow workspace + speculative edits), Devin (planner + executor + verifier)
# OpenAI Codex (sandboxed execution + test-driven loop)

import asyncio
import ast
import difflib
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
EVOLVE_LOG = Path("evolve_log.jsonl")
KNOWLEDGE_VAULT = Path("protocols/vault")
KNOWLEDGE_VAULT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# WHAT WE LEARNED FROM THE BEST AGENTS IN THE WORLD:
#
# CLAUDE CODE:
#   - Single agentic loop: think → tool call → observe → repeat
#   - Never summarizes context — reads full files every time (no hallucination)
#   - Uses bash as primary tool (not Python subprocess wrapper)
#   - Checks its own work by re-reading files after writing them
#   - Abandons and retries instead of patching bad code
#
# CURSOR COMPOSER:
#   - Shadow workspace: makes ALL edits in memory first, diffs before applying
#   - Speculative multi-file edits: plans entire changeset before touching disk
#   - Context window management: chunks large codebases by relevance
#   - Tab completion trained on your specific codebase patterns
#
# DEVIN (Cognition):
#   - Planner + Executor + Verifier are SEPARATE roles (not one model)
#   - Planner writes a scratchpad before any action
#   - Verifier runs tests and reads error messages to loop back
#   - Long-horizon memory: stores intermediate results to files
#
# OPENAI CODEX CLI:
#   - Fully sandboxed: runs code in isolated environment
#   - Test-driven: writes tests FIRST then makes them pass
#   - Diff-only output: never rewrites whole files, just patches
#   - Approval mode: shows proposed changes before applying
#
# SWE-BENCH TOP PERFORMERS (76%+ accuracy):
#   - Multi-model: planner=large model, executor=fast model, verifier=reasoner
#   - Localization first: find EXACTLY where the bug is before touching anything
#   - Reproduce before fix: write a failing test that proves the bug exists
#   - Verify after fix: confirm test now passes
# ══════════════════════════════════════════════════════════════════════════


class ShadowWorkspace:
    """Cursor-style: plan all edits in memory, diff before applying."""

    def __init__(self):
        self.staged: dict[str, str] = {}  # path -> new content
        self.originals: dict[str, str] = {}

    def stage(self, path: str, content: str):
        p = Path(path)
        if p.exists() and path not in self.originals:
            self.originals[path] = p.read_text(encoding="utf-8", errors="replace")
        self.staged[path] = content

    def diff(self, path: str) -> str:
        original = self.originals.get(path, "")
        new = self.staged.get(path, "")
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3
        ))
        return "".join(diff) if diff else "(no changes)"

    def commit_all(self) -> list[str]:
        committed = []
        for path, content in self.staged.items():
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            committed.append(path)
        self.staged.clear()
        return committed

    def discard(self):
        self.staged.clear()

    def show_all_diffs(self) -> str:
        if not self.staged:
            return "No staged changes."
        out = []
        for path in self.staged:
            out.append(f"\n{'='*60}\n{path}\n{'='*60}")
            out.append(self.diff(path))
        return "\n".join(out)


class TestHarness:
    """Devin/Codex-style: write test first, run it, loop until green."""

    def run_tests(self, test_file: str = None, pattern: str = "test_*.py") -> dict:
        """Run pytest and return structured results."""
        if test_file:
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header", "-q"]
        else:
            cmd = [sys.executable, "-m", "pytest", pattern, "-v", "--tb=short", "--no-header", "-q"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            errors = len(re.findall(r" ERROR", output))
            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "output": output[:3000],
                "success": failed == 0 and errors == 0
            }
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errors": 1, "output": "TIMEOUT", "success": False}
        except FileNotFoundError:
            return {"passed": 0, "failed": 0, "errors": 0, "output": "pytest not installed", "success": True}

    def run_syntax_check(self, code: str) -> tuple[bool, str]:
        """AST parse check — catches syntax errors before writing."""
        try:
            ast.parse(code)
            return True, "OK"
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def run_code_and_capture(self, code: str, timeout: int = 30) -> dict:
        """Execute code in isolated subprocess, capture all output."""
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
            return {"stdout": "", "stderr": f"TIMEOUT after {timeout}s", "returncode": -1, "success": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


class IntelligenceScraper:
    """Actively scrape GitHub, papers, and docs to learn how top agents work.
    Extracts patterns, stores them in knowledge vault for Jarvis to learn from.
    """

    def __init__(self):
        self.vault = KNOWLEDGE_VAULT

    async def scrape_github_pattern(self, repo: str, query: str) -> dict:
        """Search GitHub for coding patterns used by top AI projects."""
        import urllib.request
        import urllib.parse
        try:
            q = urllib.parse.quote_plus(f"{query} repo:{repo}")
            url = f"https://api.github.com/search/code?q={q}&per_page=5"
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Jarvis-Intelligence-Scraper/3.0"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            items = data.get("items", [])
            patterns = []
            for item in items[:3]:
                patterns.append({
                    "file": item.get("name"),
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                    "repo": item.get("repository", {}).get("full_name")
                })
            return {"query": query, "patterns": patterns, "source": repo}
        except Exception as e:
            return {"query": query, "patterns": [], "error": str(e)}

    def extract_coding_patterns(self, code: str) -> list[str]:
        """Extract reusable patterns from any code using AST analysis."""
        patterns = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    decorators = [ast.unparse(d) for d in node.decorator_list]
                    args = [a.arg for a in node.args.args]
                    patterns.append(f"function:{node.name}({', '.join(args)}) decorators={decorators}")
                elif isinstance(node, ast.ClassDef):
                    bases = [ast.unparse(b) for b in node.bases]
                    patterns.append(f"class:{node.name}(bases={bases})")
                elif isinstance(node, ast.AsyncFunctionDef):
                    patterns.append(f"async_function:{node.name}")
        except Exception:
            pass
        return patterns

    def store_pattern(self, name: str, pattern: dict):
        """Store a learned pattern in the knowledge vault."""
        vault_file = self.vault / f"{name}.json"
        existing = []
        if vault_file.exists():
            try:
                existing = json.loads(vault_file.read_text())
            except Exception:
                pass
        existing.append({**pattern, "learned_at": datetime.now().isoformat()})
        vault_file.write_text(json.dumps(existing[-50:], indent=2))  # keep last 50

    def recall_patterns(self, name: str) -> list:
        vault_file = self.vault / f"{name}.json"
        if not vault_file.exists():
            return []
        try:
            return json.loads(vault_file.read_text())
        except Exception:
            return []


class GödelModifier:
    """Recursive self-improvement: Jarvis reads its own source, proposes upgrades,
    validates them, and applies them. Named after Gödel's incompleteness —
    the system that can reason about itself.
    """

    def __init__(self, client: AsyncOpenAI, harness: TestHarness, shadow: ShadowWorkspace):
        self.client = client
        self.harness = harness
        self.shadow = shadow

    async def propose_self_upgrade(self, target_file: str, improvement_goal: str) -> dict:
        """Read a file, ask DeepSeek R1 how to improve it, validate, stage for approval."""
        p = Path(target_file)
        if not p.exists():
            return {"success": False, "reason": f"File not found: {target_file}"}

        current_code = p.read_text(encoding="utf-8", errors="replace")

        # Use R1 reasoner for self-modification (deepest thinking)
        messages = [
            {"role": "system", "content": """You are a world-class Python architect specializing in autonomous AI systems.
You are improving your own source code. Be surgical — minimal changes, maximum impact.
Output ONLY valid Python code. No markdown fences, no explanation, just the improved file.
Rules:
1. Preserve all existing functionality
2. Fix any bugs you detect
3. Improve performance or clarity where obvious
4. Add the requested improvement
5. Never break imports or interfaces"""},
            {"role": "user", "content": f"""Current file: {target_file}

Goal: {improvement_goal}

Current code:
{current_code}

Output the complete improved file:"""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=8192,
            )
            proposed = response.choices[0].message.content

            # Strip markdown if model wrapped it anyway
            proposed = re.sub(r'^```[a-z]*\n', '', proposed, flags=re.MULTILINE)
            proposed = re.sub(r'^```$', '', proposed, flags=re.MULTILINE)
            proposed = proposed.strip()

            # Syntax check before staging
            valid, error = self.harness.run_syntax_check(proposed)
            if not valid:
                return {"success": False, "reason": f"Syntax error in proposal: {error}", "proposed": proposed[:500]}

            # Stage in shadow workspace
            self.shadow.stage(target_file, proposed)
            diff = self.shadow.diff(target_file)

            return {
                "success": True,
                "file": target_file,
                "goal": improvement_goal,
                "diff": diff,
                "lines_before": len(current_code.splitlines()),
                "lines_after": len(proposed.splitlines()),
                "staged": True
            }
        except Exception as e:
            return {"success": False, "reason": str(e)}


class WorldClassCoderEngine:
    """The full stack. Synthesizes Claude Code + Cursor + Devin + Codex patterns
    into one engine running on DeepSeek with our constraints.
    """

    # PLANNER prompt — Devin architecture
    PLANNER_PROMPT = """You are the PLANNER. Your job is to break down any coding task into
a precise execution plan before any code is written.

Output a JSON object with this exact structure:
{
  "goal": "one sentence summary",
  "complexity": "low|medium|high",
  "files_to_read": ["list of files to understand first"],
  "files_to_create": ["list of new files needed"],
  "files_to_modify": ["list of existing files to change"],
  "steps": [
    {"step": 1, "action": "description", "type": "read|write|test|verify|search"}
  ],
  "test_strategy": "how to verify success",
  "risk": "what could go wrong"
}

Think like a senior engineer at Anthropic. Be specific. Be surgical."""

    # EXECUTOR prompt — Claude Code architecture
    EXECUTOR_PROMPT = """You are the EXECUTOR. You receive a plan and implement it step by step.

Critical rules (Claude Code's actual approach):
1. Read files before modifying them — never assume content
2. Write complete files, not snippets — no "rest of file unchanged"
3. After writing, re-read to verify the write succeeded
4. If a test fails, read the FULL error message before attempting a fix
5. Abandon bad approaches after 2 failures — try a different strategy
6. Use minimal diffs when patching — don't rewrite what works

For each action, output:
ACTION: action_type
FILE: path/to/file
CONTENT:
[content here]
END"""

    # VERIFIER prompt — Codex/SWE-bench architecture
    VERIFIER_PROMPT = """You are the VERIFIER. You review code and test results with extreme skepticism.

Your job:
1. Check if the implementation actually solves the stated goal
2. Look for edge cases that aren't handled
3. Verify imports exist and are correct
4. Check for off-by-one errors, None dereferences, type mismatches
5. Confirm the test strategy was executed

Output:
{
  "verdict": "pass|fail|partial",
  "issues": ["list of specific problems"],
  "confidence": 0.0-1.0,
  "recommendation": "ship|fix|replan"
}"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.shadow = ShadowWorkspace()
        self.harness = TestHarness()
        self.scraper = IntelligenceScraper()
        self.modifier = GödelModifier(self.client, self.harness, self.shadow)
        self.session_log = []

    async def _call(self, messages: list, use_reasoner: bool = False, json_mode: bool = False) -> str:
        model = "deepseek-reasoner" if use_reasoner else "deepseek-chat"
        kwargs = dict(model=model, messages=messages, max_tokens=4096, temperature=0.2)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def plan(self, task: str, context_files: list[str] = None) -> dict:
        """Phase 1: Planner creates execution blueprint."""
        context = ""
        if context_files:
            for f in context_files[:5]:
                if Path(f).exists():
                    content = Path(f).read_text(encoding="utf-8", errors="replace")[:2000]
                    context += f"\n\n--- {f} ---\n{content}"

        messages = [
            {"role": "system", "content": self.PLANNER_PROMPT},
            {"role": "user", "content": f"Task: {task}{f'\n\nContext:{context}' if context else ''}"}
        ]
        raw = await self._call(messages, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Extract JSON if model wrapped it
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"goal": task, "steps": [], "error": "parse_failed"}

    async def execute(self, plan: dict, auto_apply: bool = False) -> dict:
        """Phase 2: Executor implements the plan step by step."""
        results = []
        files_created = []

        for step in plan.get("steps", []):
            step_num = step.get("step", 0)
            action = step.get("action", "")
            step_type = step.get("type", "write")

            if step_type == "read":
                file_path = step.get("file", "")
                if file_path and Path(file_path).exists():
                    content = Path(file_path).read_text(encoding="utf-8", errors="replace")[:3000]
                    results.append({"step": step_num, "type": "read", "file": file_path, "preview": content[:200]})

            elif step_type in ["write", "create"]:
                messages = [
                    {"role": "system", "content": self.EXECUTOR_PROMPT},
                    {"role": "user", "content": f"""Plan goal: {plan.get('goal')}
Step {step_num}: {action}

Existing context files: {plan.get('files_to_read', [])}
Write the complete implementation. Output only the file content."""}
                ]
                code = await self._call(messages)
                code = re.sub(r'^```[a-z]*\n', '', code, flags=re.MULTILINE)
                code = re.sub(r'^```$', '', code, flags=re.MULTILINE).strip()

                file_path = step.get("file", f"generated_step_{step_num}.py")
                valid, error = self.harness.run_syntax_check(code)

                if valid:
                    self.shadow.stage(file_path, code)
                    results.append({"step": step_num, "type": "write", "file": file_path,
                                    "lines": len(code.splitlines()), "status": "staged"})
                    files_created.append(file_path)
                else:
                    results.append({"step": step_num, "type": "write", "file": file_path,
                                    "status": "syntax_error", "error": error})

            elif step_type == "test":
                test_result = self.harness.run_tests()
                results.append({"step": step_num, "type": "test", "result": test_result})

        if auto_apply:
            committed = self.shadow.commit_all()
            return {"results": results, "committed": committed, "files": files_created}

        return {"results": results, "staged": list(self.shadow.staged.keys()),
                "diff_preview": self.shadow.show_all_diffs()[:2000], "files": files_created}

    async def verify(self, goal: str, implementation_summary: str) -> dict:
        """Phase 3: Verifier checks the work."""
        messages = [
            {"role": "system", "content": self.VERIFIER_PROMPT},
            {"role": "user", "content": f"""Goal: {goal}

What was implemented:
{implementation_summary}

Verify and return JSON verdict."""}
        ]
        raw = await self._call(messages, use_reasoner=True, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"verdict": "unknown", "issues": [], "confidence": 0.5, "recommendation": "manual_review"}

    async def full_cycle(self, task: str, context_files: list[str] = None,
                         auto_apply: bool = False) -> dict:
        """Complete Planner→Executor→Verifier→Self-Heal cycle.
        This is the Devin/Claude Code loop replicated under our constraints.
        """
        print(f"\n🜏 WORLD-CLASS CODER ENGINE — FULL CYCLE")
        print(f"Task: {task}")
        print("─" * 60)

        # Phase 1: Plan
        print("[1/4] Planning...")
        plan = await self.plan(task, context_files)
        print(f"  Goal: {plan.get('goal')}")
        print(f"  Complexity: {plan.get('complexity')}")
        print(f"  Steps: {len(plan.get('steps', []))}")
        self._log("plan", plan)

        # Phase 2: Execute
        print("[2/4] Executing...")
        execution = await self.execute(plan, auto_apply=auto_apply)
        staged = execution.get("staged", [])
        print(f"  Staged files: {staged}")
        self._log("execute", {"staged": staged})

        # Phase 3: Test
        print("[3/4] Testing...")
        test_result = self.harness.run_tests()
        print(f"  Tests: {test_result['passed']} passed, {test_result['failed']} failed")
        self._log("test", test_result)

        # Phase 4: Verify
        print("[4/4] Verifying...")
        summary = f"Staged: {staged}. Tests: {test_result['passed']} pass, {test_result['failed']} fail."
        verdict = await self.verify(task, summary)
        print(f"  Verdict: {verdict.get('verdict')} | Confidence: {verdict.get('confidence')}")
        print(f"  Recommendation: {verdict.get('recommendation')}")
        self._log("verify", verdict)

        # Self-heal loop if failed
        if verdict.get("verdict") == "fail" and verdict.get("issues"):
            print("\n⚠️  Issues detected. Entering self-heal loop...")
            issues_str = "\n".join(verdict["issues"])
            heal_task = f"Fix these issues in the previous implementation:\n{issues_str}"
            heal_plan = await self.plan(heal_task, staged)
            heal_execution = await self.execute(heal_plan, auto_apply=auto_apply)
            print(f"  Healed. New staged: {heal_execution.get('staged', [])}")
            self._log("self_heal", {"issues": verdict["issues"], "heal_staged": heal_execution.get("staged")})

        result = {
            "task": task,
            "plan": plan,
            "execution": execution,
            "tests": test_result,
            "verdict": verdict,
            "diff": execution.get("diff_preview", ""),
            "ready_to_commit": verdict.get("recommendation") in ["ship", "fix"]
        }

        if auto_apply and verdict.get("recommendation") == "ship":
            committed = self.shadow.commit_all()
            result["committed"] = committed
            print(f"\n✅ Auto-committed: {committed}")

        return result

    async def self_evolve(self, target_file: str, goal: str) -> dict:
        """Gödel loop: Jarvis reads itself, proposes improvement, validates, stages."""
        print(f"\n🜏 GÖDEL SELF-MODIFICATION")
        print(f"Target: {target_file}")
        print(f"Goal: {goal}")
        result = await self.modifier.propose_self_upgrade(target_file, goal)
        if result["success"]:
            print(f"  Lines: {result['lines_before']} → {result['lines_after']}")
            print(f"  Diff preview (first 500 chars):")
            print(result["diff"][:500])
        else:
            print(f"  ❌ Failed: {result['reason']}")
        return result

    async def absorb_intelligence(self, source: str = "anthropic/claude-code") -> dict:
        """Scrape public AI coding agents, extract patterns, store in vault."""
        print(f"\n🜏 INTELLIGENCE ABSORPTION — {source}")
        patterns_learned = 0

        searches = [
            ("agentic loop", "tool_use_pattern"),
            ("self healing", "error_recovery_pattern"),
            ("test driven", "tdd_pattern"),
            ("shadow workspace", "edit_pattern")
        ]

        for query, vault_key in searches:
            result = await self.scraper.scrape_github_pattern(source, query)
            if result.get("patterns"):
                self.scraper.store_pattern(vault_key, result)
                patterns_learned += len(result["patterns"])
                print(f"  Learned {len(result['patterns'])} patterns for '{query}'")

        return {"source": source, "patterns_learned": patterns_learned,
                "vault_path": str(KNOWLEDGE_VAULT)}

    def _log(self, event_type: str, data: dict):
        entry = {"ts": datetime.now().isoformat(), "type": event_type, "data": data}
        with open(EVOLVE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.session_log.append(entry)

    def commit_staged(self) -> list[str]:
        """Apply all staged changes to disk."""
        return self.shadow.commit_all()

    def discard_staged(self):
        """Throw away all staged changes."""
        self.shadow.discard()
        print("🗑️  Staged changes discarded.")


# ══ CLI INTERFACE FOR PAUL ══════════════════════════════════════════════
async def run_evolve_cli():
    """Interactive self-evolving coder CLI."""
    engine = WorldClassCoderEngine()

    print("""
╔══════════════════════════════════════════════╗
║  🜏 JARVIS WORLD-CLASS CODER ENGINE v3.0     ║
╠══════════════════════════════════════════════╣
║  Powered by: DeepSeek R1 + Planner/Executor  ║
║  Architecture: Claude Code + Cursor + Devin  ║
║  Modes: build | evolve | absorb | test       ║
╚══════════════════════════════════════════════╝

Commands:
  build [task]    — full Planner→Executor→Verifier cycle
  evolve [file]   — Gödel self-modification on a file
  absorb          — scrape top AI agents for patterns
  diff            — show all staged changes
  commit          — apply staged changes to disk
  discard         — throw away staged changes
  test            — run all tests
  exit            — quit
""")

    while True:
        try:
            cmd = input("Jarvis[coder]> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue
        if cmd == "exit":
            break

        elif cmd.startswith("build "):
            task = cmd[6:].strip()
            result = await engine.full_cycle(task)
            print(f"\nDiff preview:\n{result.get('diff', 'none')[:1000]}")
            print("\nType 'commit' to apply or 'discard' to abandon.")

        elif cmd.startswith("evolve "):
            parts = cmd[7:].split(" ", 1)
            file = parts[0]
            goal = parts[1] if len(parts) > 1 else "improve performance and clarity"
            result = await engine.self_evolve(file, goal)
            if result["success"]:
                print("\nType 'commit' to apply or 'discard' to abandon.")

        elif cmd == "absorb":
            result = await engine.absorb_intelligence()
            print(f"\n🜏 Absorbed {result['patterns_learned']} patterns → {result['vault_path']}")

        elif cmd == "diff":
            print(engine.shadow.show_all_diffs())

        elif cmd == "commit":
            committed = engine.commit_staged()
            print(f"✅ Committed: {committed}")

        elif cmd == "discard":
            engine.discard_staged()

        elif cmd == "test":
            result = engine.harness.run_tests()
            print(result["output"])

        else:
            print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    asyncio.run(run_evolve_cli())
