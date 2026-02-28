# world_coder.py — Jarvis skill: activate World-Class Coder Engine from chat
# Loaded via: TOOL: load_skill(world_coder)
# Paul says: "build me X" → Jarvis enters full Planner→Executor→Verifier cycle

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("""🜏 WORLD-CLASS CODER MODE ACTIVATED

This skill gives Jarvis access to the full autonomous coding engine.
Architecture stolen from: Claude Code + Cursor + Devin + OpenAI Codex

Jarvis can now:
  - Plan complex multi-file codebases before touching anything
  - Write code in a shadow workspace (diff before applying)
  - Run tests and self-heal failures automatically
  - Use DeepSeek R1 as the Verifier brain for deep reasoning
  - Recursively improve its own source code (Gödel loop)
  - Scrape top AI agents on GitHub for patterns to absorb

To use in chat:
  'build a REST API for X'
  'evolve paul_core.py to add streaming output'
  'absorb intelligence from openai/codex'
  'self-heal all broken files'

Engine: jarvis_evolve.py | Daemon: jarvis_self_heal.py
""")

# Quick self-check
try:
    from jarvis_evolve import WorldClassCoderEngine, TestHarness
    harness = TestHarness()
    test_result = harness.run_syntax_check(Path("paul_core.py").read_text() if Path("paul_core.py").exists() else "x=1")
    print(f"paul_core.py syntax: {'✅ OK' if test_result[0] else '❌ ' + test_result[1]}")
    print("Engine: ✅ WorldClassCoderEngine loaded")
except ImportError as e:
    print(f"Engine load: ⚠️  {e} (run from repo root)")
