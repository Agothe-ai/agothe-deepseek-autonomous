# jarvis_github_watcher.py — Jarvis GitHub Live Intelligence v5.0
# Watches Paul's GitHub repos in real time.
# Every push → DeepSeek R1 code review → spoken feedback in <30 seconds.
# No human reviewer needed. Jarvis IS the code review.

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from openai import AsyncOpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-71b52b116f3c432d8e7bfeeec42edf4c")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "gtsgob")
POLL_INTERVAL = 30          # seconds between GitHub checks
MAX_DIFF_CHARS = 6000       # max diff to send to R1
WATCH_LOG = Path("github_watch_log.jsonl")

# Repos to watch — auto-discovers all of Paul's repos if empty
WATCH_REPOS = [
    "gtsgob/agothe-deepseek-autonomous",
]


class GitHubPoller:
    """Polls GitHub API for new commits, PRs, and push events."""

    def __init__(self, token: str = GITHUB_TOKEN, username: str = GITHUB_USERNAME):
        self.token = token
        self.username = username
        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Jarvis-GitHub-Watcher/5.0"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        # Track last seen commit per repo
        self.last_seen: dict[str, str] = {}
        self.last_seen_file = Path("github_last_seen.json")
        self._load_last_seen()

    def _load_last_seen(self):
        if self.last_seen_file.exists():
            try:
                self.last_seen = json.loads(self.last_seen_file.read_text())
            except Exception:
                pass

    def _save_last_seen(self):
        self.last_seen_file.write_text(json.dumps(self.last_seen, indent=2))

    async def _get(self, url: str) -> dict | list | None:
        """Make authenticated GitHub API request."""
        import urllib.request
        import urllib.error
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403:
                print(f"  ⚠️  GitHub rate limit hit. Waiting...")
                await asyncio.sleep(60)
                return None
            return None
        except Exception as e:
            print(f"  GitHub API error: {e}")
            return None

    async def get_latest_commits(self, repo: str, limit: int = 5) -> list[dict]:
        """Get most recent commits for a repo."""
        url = f"https://api.github.com/repos/{repo}/commits?per_page={limit}"
        data = await self._get(url)
        if not isinstance(data, list):
            return []
        return data

    async def get_commit_diff(self, repo: str, sha: str) -> str:
        """Get the full diff for a specific commit."""
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        data = await self._get(url)
        if not data:
            return ""

        files = data.get("files", [])
        diff_parts = []

        for f in files[:10]:  # max 10 files per commit
            filename = f.get("filename", "")
            patch = f.get("patch", "")
            status = f.get("status", "")
            additions = f.get("additions", 0)
            deletions = f.get("deletions", 0)

            if patch:
                diff_parts.append(
                    f"--- {filename} ({status}: +{additions}/-{deletions}) ---\n{patch}"
                )

        full_diff = "\n\n".join(diff_parts)
        return full_diff[:MAX_DIFF_CHARS]

    async def get_open_prs(self, repo: str) -> list[dict]:
        """Get open pull requests."""
        url = f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=10"
        data = await self._get(url)
        return data if isinstance(data, list) else []

    async def get_pr_diff(self, repo: str, pr_number: int) -> str:
        """Get the diff for a PR."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=20"
        data = await self._get(url)
        if not isinstance(data, list):
            return ""

        diff_parts = []
        for f in data[:10]:
            filename = f.get("filename", "")
            patch = f.get("patch", "")
            status = f.get("status", "")
            if patch:
                diff_parts.append(f"--- {filename} ({status}) ---\n{patch}")

        return "\n\n".join(diff_parts)[:MAX_DIFF_CHARS]

    async def discover_repos(self) -> list[str]:
        """Auto-discover all of Paul's public repos."""
        url = f"https://api.github.com/users/{self.username}/repos?per_page=30&sort=pushed"
        data = await self._get(url)
        if not isinstance(data, list):
            return WATCH_REPOS
        return [r["full_name"] for r in data if not r.get("archived", False)]

    async def poll_new_commits(self, repos: list[str]) -> list[dict]:
        """Check all repos for new commits since last check. Returns new ones."""
        new_commits = []

        for repo in repos:
            commits = await self.get_latest_commits(repo, limit=3)
            if not commits:
                continue

            latest_sha = commits[0].get("sha", "")
            last_sha = self.last_seen.get(repo, "")

            if latest_sha and latest_sha != last_sha:
                # Find all new commits (not seen before)
                for commit in commits:
                    sha = commit.get("sha", "")
                    if sha == last_sha:
                        break
                    commit_data = {
                        "repo": repo,
                        "sha": sha,
                        "message": commit.get("commit", {}).get("message", "")[:200],
                        "author": commit.get("commit", {}).get("author", {}).get("name", ""),
                        "timestamp": commit.get("commit", {}).get("author", {}).get("date", ""),
                        "url": commit.get("html_url", "")
                    }
                    new_commits.append(commit_data)

                self.last_seen[repo] = latest_sha

        if new_commits:
            self._save_last_seen()

        return new_commits


class CommitReviewer:
    """Uses DeepSeek R1 to review code diffs like a senior engineer."""

    REVIEW_PROMPT = """You are Jarvis — an expert code reviewer with the skills of a senior engineer at Anthropic, Google, and Stripe combined.

You are reviewing Paul's latest commit. Be direct. Be specific. Be useful.
Paul is building toward world-class code. Treat him like a peer, not a student.

For each review, structure your response as:
1. VERDICT: one of SHIP_IT / MINOR_ISSUES / NEEDS_WORK / CRITICAL
2. SUMMARY: 1-2 sentences on what this commit does
3. STRENGTHS: what's good (be specific, max 3 points)
4. ISSUES: concrete problems if any (max 3, ranked by severity)
5. ONE_THING: the single most important improvement Paul should make next

Keep the total response under 300 words. Paul will hear this spoken aloud.
No markdown headers, no bullet symbols — plain text, punchy sentences."""

    PR_REVIEW_PROMPT = """You are Jarvis reviewing a pull request for Paul.
Give a thorough but spoken-friendly review. Be a brilliant, direct senior engineer.

Structure:
1. PR_VERDICT: APPROVE / REQUEST_CHANGES / COMMENT
2. WHAT_IT_DOES: one sentence
3. CODE_QUALITY: score 1-10, one sentence explanation
4. TOP_ISSUE: the most important thing to fix (or 'None' if clean)
5. SHIP_RECOMMENDATION: should this merge? Why in one sentence.

Max 250 words. Plain text for speech."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.review_count = 0

    async def review_commit(self, commit: dict, diff: str) -> dict:
        """Review a commit diff using R1."""
        if not diff:
            return {
                "verdict": "NO_DIFF",
                "summary": f"Commit by {commit['author']}: {commit['message'][:100]}",
                "spoken": f"New commit pushed: {commit['message'][:100]}. No diff available to review."
            }

        messages = [
            {"role": "system", "content": self.REVIEW_PROMPT},
            {"role": "user", "content": f"""Repo: {commit['repo']}
Commit: {commit['sha'][:8]}
Author: {commit['author']}
Message: {commit['message']}

Diff:
{diff}

Review this commit:"""}
        ]

        try:
            # R1 for deep code review — this is the verifier brain
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=1024
            )
            review_text = response.choices[0].message.content
            self.review_count += 1

            # Extract verdict
            verdict = "UNKNOWN"
            for v in ["SHIP_IT", "MINOR_ISSUES", "NEEDS_WORK", "CRITICAL"]:
                if v in review_text.upper():
                    verdict = v
                    break

            # Build spoken version — natural, not robotic
            spoken = f"New commit in {commit['repo'].split('/')[-1]}. " + review_text[:500]

            result = {
                "verdict": verdict,
                "review": review_text,
                "spoken": spoken,
                "commit": commit,
                "reviewed_at": datetime.now().isoformat(),
                "review_num": self.review_count
            }

            # Log it
            with open(WATCH_LOG, "a") as f:
                f.write(json.dumps({
                    "ts": result["reviewed_at"],
                    "type": "commit_review",
                    "repo": commit["repo"],
                    "sha": commit["sha"][:8],
                    "verdict": verdict,
                    "review_num": self.review_count
                }) + "\n")

            return result

        except Exception as e:
            return {
                "verdict": "ERROR",
                "review": str(e),
                "spoken": f"I tried to review the commit but hit an error: {str(e)[:100]}",
                "commit": commit
            }

    async def review_pr(self, repo: str, pr: dict, diff: str) -> dict:
        """Review a pull request."""
        messages = [
            {"role": "system", "content": self.PR_REVIEW_PROMPT},
            {"role": "user", "content": f"""Repo: {repo}
PR #{pr.get('number')}: {pr.get('title', '')}
Author: {pr.get('user', {}).get('login', '')}
Description: {pr.get('body', '')[:500]}

Diff:
{diff}

Review this PR:"""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=800
            )
            review_text = response.choices[0].message.content

            verdict = "COMMENT"
            if "APPROVE" in review_text.upper():
                verdict = "APPROVE"
            elif "REQUEST_CHANGES" in review_text.upper():
                verdict = "REQUEST_CHANGES"

            return {
                "verdict": verdict,
                "review": review_text,
                "spoken": f"Pull request review for {pr.get('title', 'untitled')}. " + review_text[:400],
                "pr_number": pr.get("number"),
                "pr_url": pr.get("html_url", "")
            }
        except Exception as e:
            return {"verdict": "ERROR", "review": str(e), "spoken": f"PR review error: {e}"}


class GitHubWatcherDaemon:
    """The full live watcher. Polls GitHub every 30s.
    New commit detected → R1 review → spoken feedback.
    This is the world's first AI code reviewer that speaks.
    """

    def __init__(self, voice_enabled: bool = True):
        self.poller = GitHubPoller()
        self.reviewer = CommitReviewer()
        self.voice_enabled = voice_enabled
        self.voice = None
        self.running = False
        self.repos_watched: list[str] = []
        self.review_count = 0
        self.start_time = datetime.now()

        if voice_enabled:
            try:
                from jarvis_voice import JarvisVoice, VoicePersonalityEngine
                self.voice = JarvisVoice()
                self.personality = VoicePersonalityEngine(self.voice)
            except ImportError:
                print("  Voice not available (jarvis_voice.py not found)")
                self.voice_enabled = False

    def _speak(self, text: str):
        """Speak if voice available, always print."""
        print(f"\n🜏 Jarvis: {text}")
        if self.voice_enabled and self.voice:
            self.voice.speak(text)

    def _print_review(self, review: dict):
        """Print a formatted code review to terminal."""
        verdict = review.get("verdict", "?")
        verdict_icons = {
            "SHIP_IT": "✅",
            "MINOR_ISSUES": "⚠️",
            "NEEDS_WORK": "🔧",
            "CRITICAL": "🚨",
            "APPROVE": "✅",
            "REQUEST_CHANGES": "🔧",
            "COMMENT": "💬"
        }
        icon = verdict_icons.get(verdict, "🜏")
        commit = review.get("commit", {})

        print(f"\n{'='*60}")
        print(f"{icon} VERDICT: {verdict}")
        print(f"Repo:    {commit.get('repo', '')}")
        print(f"Commit:  {commit.get('sha', '')[:8]} — {commit.get('message', '')[:60]}")
        print(f"Author:  {commit.get('author', '')}")
        print(f"URL:     {commit.get('url', '')}")
        print(f"{'='*60}")
        print(review.get("review", "")[:1000])
        print(f"{'='*60}\n")

    async def run_forever(self, auto_discover: bool = True):
        """Main watch loop."""
        self.running = True

        # Discover repos
        if auto_discover:
            print("  Discovering repos...")
            self.repos_watched = await self.poller.discover_repos()
        else:
            self.repos_watched = WATCH_REPOS

        print(f"\n🜏 GITHUB LIVE WATCHER ACTIVE")
        print(f"   Watching {len(self.repos_watched)} repos for {GITHUB_USERNAME}")
        print(f"   Check interval: {POLL_INTERVAL}s")
        print(f"   Voice feedback: {'ON' if self.voice_enabled else 'OFF (text only)'}")
        print(f"   Repos: {', '.join(self.repos_watched[:5])}")
        print(f"   Log: {WATCH_LOG}")
        print(f"   Ctrl+C to stop\n")

        # Initial state — don't review already-existing commits on first run
        print("  Initializing baseline...")
        await self.poller.poll_new_commits(self.repos_watched)
        print("  Baseline set. Watching for new pushes...\n")

        if self.voice_enabled:
            self._speak(f"GitHub watcher online. Monitoring {len(self.repos_watched)} repos. I'll review every push within 30 seconds.")

        while self.running:
            try:
                await asyncio.sleep(POLL_INTERVAL)

                # Check for new commits
                new_commits = await self.poller.poll_new_commits(self.repos_watched)

                for commit in new_commits:
                    print(f"  🔔 New commit: {commit['repo']} — {commit['message'][:60]}")

                    # Get the diff
                    diff = await self.poller.get_commit_diff(commit["repo"], commit["sha"])

                    # R1 review
                    print(f"  🜏 Reviewing with R1...")
                    review = await self.reviewer.review_commit(commit, diff)
                    self.review_count += 1

                    # Display
                    self._print_review(review)

                    # Speak the verdict
                    self._speak(review["spoken"])

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"  Watcher error: {e}")
                await asyncio.sleep(POLL_INTERVAL)

        uptime = (datetime.now() - self.start_time).seconds
        print(f"\n🜏 GitHub watcher stopped. Uptime: {uptime}s | Reviews: {self.review_count}")
        if self.voice_enabled:
            self._speak("GitHub watcher offline.")

    async def review_repo_now(self, repo: str) -> list[dict]:
        """On-demand: review the last N commits of any repo immediately."""
        print(f"\n🜏 ON-DEMAND REVIEW: {repo}")
        commits = await self.poller.get_latest_commits(repo, limit=3)
        reviews = []

        for commit in commits:
            diff = await self.poller.get_commit_diff(repo, commit["sha"])
            review = await self.reviewer.review_commit(
                {
                    "repo": repo,
                    "sha": commit["sha"],
                    "message": commit.get("commit", {}).get("message", "")[:200],
                    "author": commit.get("commit", {}).get("author", {}).get("name", ""),
                    "url": commit.get("html_url", "")
                },
                diff
            )
            self._print_review(review)
            if self.voice_enabled:
                self._speak(review["spoken"][:300])
            reviews.append(review)

        return reviews

    async def review_prs_now(self, repo: str) -> list[dict]:
        """On-demand: review all open PRs for a repo."""
        print(f"\n🜏 PR REVIEW: {repo}")
        prs = await self.poller.get_open_prs(repo)
        reviews = []

        if not prs:
            msg = f"No open pull requests in {repo}."
            print(f"  {msg}")
            if self.voice_enabled:
                self._speak(msg)
            return []

        for pr in prs[:3]:
            diff = await self.poller.get_pr_diff(repo, pr["number"])
            review = await self.reviewer.review_pr(repo, pr, diff)
            print(f"\n  PR #{pr['number']}: {pr['title']}")
            print(f"  Verdict: {review['verdict']}")
            print(f"  {review['review'][:600]}")
            if self.voice_enabled:
                self._speak(review["spoken"][:300])
            reviews.append(review)

        return reviews

    def status(self) -> dict:
        uptime = (datetime.now() - self.start_time).seconds
        return {
            "running": self.running,
            "repos_watched": len(self.repos_watched),
            "reviews_performed": self.review_count,
            "uptime_seconds": uptime,
            "voice": self.voice_enabled,
            "log": str(WATCH_LOG)
        }


# ══ CLI ══════════════════════════════════════════════════════════
async def run_watcher_cli():
    """Interactive GitHub watcher CLI."""
    print("""
╔══════════════════════════════════════════════════╗
║  🜏 JARVIS GITHUB LIVE WATCHER v5.0              ║
╠══════════════════════════════════════════════════╣
║  Every push → R1 review → spoken in <30s        ║
║  Commands: watch | review [repo] | prs [repo]   ║
║             status | exit                        ║
╚══════════════════════════════════════════════════╝
""")

    # Check GitHub token
    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN not set. Rate limited to 60 req/hour.")
        print("   Set it: set GITHUB_TOKEN=ghp_yourtoken")
        print("   Get one: https://github.com/settings/tokens\n")

    daemon = GitHubWatcherDaemon(voice_enabled=True)

    while True:
        try:
            cmd = input("Jarvis[github]> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue

        if cmd == "exit":
            break

        elif cmd == "watch":
            await daemon.run_forever(auto_discover=True)

        elif cmd.startswith("review"):
            parts = cmd.split()
            repo = parts[1] if len(parts) > 1 else f"{GITHUB_USERNAME}/agothe-deepseek-autonomous"
            if "/" not in repo:
                repo = f"{GITHUB_USERNAME}/{repo}"
            await daemon.review_repo_now(repo)

        elif cmd.startswith("prs"):
            parts = cmd.split()
            repo = parts[1] if len(parts) > 1 else f"{GITHUB_USERNAME}/agothe-deepseek-autonomous"
            if "/" not in repo:
                repo = f"{GITHUB_USERNAME}/{repo}"
            await daemon.review_prs_now(repo)

        elif cmd == "status":
            s = daemon.status()
            for k, v in s.items():
                print(f"  {k}: {v}")

        elif cmd == "discover":
            repos = await daemon.poller.discover_repos()
            print(f"Found {len(repos)} repos:")
            for r in repos:
                print(f"  {r}")

        else:
            print(f"Unknown: {cmd}")


if __name__ == "__main__":
    asyncio.run(run_watcher_cli())
