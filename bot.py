"""Single-file AI community companion.

This file intentionally keeps the whole system in one place as requested,
while covering the main major features from the project specification:
- name-trigger detection
- introduction detection
- SQLite-backed identity + memory
- call mode and reply detection
- community/project memory stubs
- challenge system
- leaderboard/progress
- Python news and tool placeholders
- voice and permissions scaffolding

It is messy by design, but complete enough to run as a single-bot codebase.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:
        return None

try:
    import discord
    from discord.ext import commands
except Exception:  # pragma: no cover
    discord = None
    commands = None


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "movie_data.db"
AI_NAME = os.getenv("AI_NAME", "Movie")
POSTGRES_DSN = (
    os.getenv("POSTGRES_DSN")
    or (
        f"dbname={os.getenv('POSTGRES_DB', 'movie_db')} "
        f"user={os.getenv('POSTGRES_USER', 'movie_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', '')} "
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')}"
    )
    if os.getenv("POSTGRES_DB") or os.getenv("POSTGRES_USER") or os.getenv("POSTGRES_HOST")
    else None
)
SYSTEM_PROMPT = (
    "You are Movie, a calm, technically precise, direct AI companion for a Python community. "
    "You are strong in Python, debugging, software design, async work, and helping people move projects forward. "
    "Stay grounded, practical, and natural. Do not be overly enthusiastic, theatrical, or chatty. "
    "Respond like a reliable teammate who is clear, honest, and useful."
)


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize_name(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


# ----------------------------
# Name trigger and intro logic
# ----------------------------
def name_trigger_score(text: str, name: str = AI_NAME) -> float:
    """Confidence scoring for direct address detection.
    Matches the documented project rules: high / ambiguous / low bands.
    """
    if not text or not isinstance(text, str):
        return 0.0

    snippet = text.strip()
    if not snippet:
        return 0.0

    lower = snippet.lower()
    name_lower = name.lower()
    score = 0.0

    if name_lower in lower:
        score += 0.30

    if re.search(rf"^(?:hey|yo|ok|okay)\s+{re.escape(name_lower)}\b|^{re.escape(name_lower)}\b", lower):
        score += 0.45

    if re.search(rf"{re.escape(name_lower)}\s*[?!]\s*$", lower):
        score += 0.35

    if "?" in snippet:
        score += 0.10

    if len(snippet.split()) <= 7:
        score += 0.30

    article_pattern = rf"(?:^|\s)(?:a|the|this)\b(?:\s+\w+){{0,2}}\s+{re.escape(name_lower)}\b"
    if re.search(article_pattern, lower):
        score -= 0.55

    watch_pattern = rf"(?:watch|watching|watched|see|seen|rent|stream|view)\b(?:\s+\w+){{0,3}}\s+{re.escape(name_lower)}\b"
    if re.search(watch_pattern, lower):
        score -= 0.55

    return max(0.0, min(1.0, score))


INTRODUCTION_PATTERNS = [
    (r"(?i:(?:this\s+is|meet|introduce|introducing))\s+([A-Z][a-zA-Z'-]+)", "third_party"),
    (r"(?i:(?:i\s*am|i'm|my\s+name\s+is|call\s+me))\s+([A-Z][a-zA-Z'-]+)", "self"),
    (r"(?i:(?:[A-Z][a-zA-Z'-]+)\s+is\s+(?:joining|new\s+here|just\s+joined))", "third_party"),
    (r"(?i:(?:i'd\s+like\s+you\s+all\s+to\s+meet|everyone\s+to\s+meet))\s+([A-Z][a-zA-Z'-]+)", "third_party"),
]

STOPWORDS = {
    "good", "fine", "sorry", "sure", "ok", "okay", "back", "here",
    "done", "not", "just", "still", "also", "new", "trying"
}


def detect_introductions(text: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    if not text:
        return results

    seen = set()
    for pattern, intro_type in INTRODUCTION_PATTERNS:
        for match in re.finditer(pattern, text):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if not candidate[0].isupper():
                continue
            if candidate.lower() in STOPWORDS:
                continue
            key = (candidate, intro_type)
            if key not in seen:
                seen.add(key)
                results.append(key)
    return results


def score_memory_importance(message: str) -> str:
    """Fast-pass memory evaluation: discard / keep / ambiguous."""
    text = message.strip().lower()
    if not text:
        return "discard"

    signals = [
        "i'm building",
        "i am building",
        "my project is",
        "current project",
        "i plan to",
        "i want to",
        "i am learning",
        "i prefer",
        "i like",
        "i hate",
        "i need",
        "i have a problem with",
        "this is my goal",
        "we decided",
        "we agreed",
        "my name is",
        "this is my",
        "i live in",
        "i work in",
        "i am allergic",
        "i use",
        "python level",
        "discord bot",
        "asyncio",
        "fastapi",
        "flask",
    ]

    if any(sig in text for sig in signals):
        return "keep"

    if re.search(r"\b(\d+|today|tomorrow|next week|next month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text):
        return "ambiguous"

    if re.search(r"\b(i am|i'm|my name is|call me|this is)\b", text):
        return "keep"

    return "discard"


# ----------------------------
# Database and storage
# ----------------------------
class MovieDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.postgres_enabled = bool(POSTGRES_DSN and psycopg is not None)
        self.postgres_dsn = POSTGRES_DSN
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def get_postgres_connection(self):
        if not self.postgres_enabled or not self.postgres_dsn:
            return None
        try:
            return psycopg.connect(self.postgres_dsn)
        except Exception:
            return None

    def init_db(self) -> None:
        conn = self.get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    discord_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    python_level TEXT DEFAULT 'beginner',
                    xp INTEGER DEFAULT 0,
                    first_seen TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS introductions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    introduced_name TEXT NOT NULL,
                    introduced_by_id TEXT,
                    introduced_by_display_name TEXT NOT NULL,
                    intro_type TEXT NOT NULL,
                    raw_text TEXT,
                    linked_discord_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (linked_discord_id) REFERENCES users(discord_id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    scope TEXT NOT NULL DEFAULT 'private',
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    importance TEXT DEFAULT 'discard',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    owner_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS project_members (
                    project_id INTEGER,
                    user_id TEXT,
                    role TEXT,
                    PRIMARY KEY(project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    difficulty TEXT,
                    xp INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    challenge_id INTEGER,
                    code TEXT,
                    score REAL DEFAULT 0,
                    feedback TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS progress (
                    user_id TEXT PRIMARY KEY,
                    level INTEGER DEFAULT 1,
                    xp INTEGER DEFAULT 0,
                    streak INTEGER DEFAULT 0,
                    challenges_completed INTEGER DEFAULT 0,
                    projects_completed INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

        if self.postgres_enabled:
            pg = self.get_postgres_connection()
            if pg is not None:
                try:
                    with pg.cursor() as cur:
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS memory_archive (
                                id BIGSERIAL PRIMARY KEY,
                                user_id TEXT,
                                scope TEXT DEFAULT 'private',
                                memory_type TEXT DEFAULT 'fact',
                                content TEXT NOT NULL,
                                importance TEXT DEFAULT 'discard',
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            );
                            """
                        )
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS project_archive (
                                id BIGSERIAL PRIMARY KEY,
                                name TEXT NOT NULL,
                                owner_id TEXT,
                                status TEXT DEFAULT 'active',
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            );
                            """
                        )
                    pg.commit()
                finally:
                    pg.close()

    def upsert_user(self, discord_id: str, display_name: str) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO users (discord_id, display_name)
                VALUES (?, ?)
                ON CONFLICT(discord_id) DO UPDATE SET display_name = excluded.display_name
                """,
                (discord_id, display_name),
            )
            conn.commit()

            conn.execute(
                "INSERT OR IGNORE INTO progress (user_id, level, xp, streak, challenges_completed, projects_completed) VALUES (?, 1, 0, 0, 0, 0)",
                (discord_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def resolve_pending_introductions(self, discord_id: str, display_name: str) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                UPDATE introductions
                SET linked_discord_id = ?
                WHERE linked_discord_id IS NULL
                  AND lower(introduced_name) = lower(?)
                """,
                (discord_id, display_name),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_introduction(self, introduced_name: str, introduced_by_id: Optional[str], introduced_by_display_name: str, intro_type: str, raw_text: Optional[str] = None) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO introductions (
                    introduced_name,
                    introduced_by_id,
                    introduced_by_display_name,
                    intro_type,
                    raw_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (introduced_name, introduced_by_id, introduced_by_display_name, intro_type, raw_text),
            )
            conn.commit()
        finally:
            conn.close()

    def add_memory(self, user_id: str, content: str, scope: str = "private", memory_type: str = "fact", importance: str = "discard") -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                "INSERT INTO memories (user_id, scope, memory_type, content, importance) VALUES (?, ?, ?, ?, ?)",
                (user_id, scope, memory_type, content, importance),
            )
            conn.commit()
        finally:
            conn.close()

        if self.postgres_enabled:
            pg = self.get_postgres_connection()
            if pg is not None:
                try:
                    with pg.cursor() as cur:
                        cur.execute(
                            "INSERT INTO memory_archive (user_id, scope, memory_type, content, importance) VALUES (%s, %s, %s, %s, %s)",
                            (user_id, scope, memory_type, content, importance),
                        )
                    pg.commit()
                finally:
                    pg.close()

    def get_recent_memories(self, user_id: str, limit: int = 6) -> List[str]:
        conn = self.get_connection()
        try:
            rows = conn.execute(
                "SELECT content FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [row["content"] for row in rows]
        finally:
            conn.close()

    def get_public_memories(self, limit: int = 10) -> List[str]:
        conn = self.get_connection()
        try:
            rows = conn.execute(
                "SELECT content FROM memories WHERE scope = 'public' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [row["content"] for row in rows]
        finally:
            conn.close()

    def create_project(self, name: str, owner_id: str) -> int:
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                "INSERT INTO projects (name, owner_id) VALUES (?, ?)",
                (name, owner_id),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def add_project_member(self, project_id: int, user_id: str, role: str = "member") -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO project_members (project_id, user_id, role) VALUES (?, ?, ?)",
                (project_id, user_id, role),
            )
            conn.commit()
        finally:
            conn.close()

    def get_projects(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def add_challenge(self, name: str, description: str, difficulty: str, xp: int = 100) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                "INSERT INTO challenges (name, description, difficulty, xp) VALUES (?, ?, ?, ?)",
                (name, description, difficulty, xp),
            )
            conn.commit()
        finally:
            conn.close()

    def submit_challenge(self, user_id: str, challenge_id: int, code: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            challenge = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
            if not challenge:
                return None

            score = 0.0
            feedback = "Not bad. Review the logic and edge cases."
            if "def" in code or "class" in code:
                score += 40
            if len(code) > 50:
                score += 20
            if "return" in code:
                score += 20
            if "try" in code or "except" in code:
                score += 10
            if "async" in code or "await" in code:
                score += 10

            score = min(score, 100)
            conn.execute(
                "INSERT INTO submissions (user_id, challenge_id, code, score, feedback) VALUES (?, ?, ?, ?, ?)",
                (user_id, challenge_id, code, score, feedback),
            )
            conn.commit()

            existing = conn.execute("SELECT xp, challenges_completed FROM progress WHERE user_id = ?", (user_id,)).fetchone()
            if existing:
                new_xp = int(existing["xp"]) + int(challenge["xp"])
                completed = int(existing["challenges_completed"]) + 1
                conn.execute(
                    "UPDATE progress SET xp = ?, challenges_completed = ?, updated_at = ? WHERE user_id = ?",
                    (new_xp, completed, now_iso(), user_id),
                )
                conn.commit()

            return {"challenge": dict(challenge), "score": score, "feedback": feedback}
        finally:
            conn.close()

    def get_progress(self, user_id: str) -> Dict[str, Any]:
        conn = self.get_connection()
        try:
            row = conn.execute("SELECT * FROM progress WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row) if row else {"level": 1, "xp": 0, "streak": 0, "challenges_completed": 0, "projects_completed": 0}
        finally:
            conn.close()

    def leaderboard(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            rows = conn.execute(
                "SELECT p.user_id, u.display_name, p.xp, p.level, p.challenges_completed FROM progress p JOIN users u ON u.discord_id = p.user_id ORDER BY p.xp DESC LIMIT 10"
            ).fetchall()
            output = []
            for i, row in enumerate(rows, start=1):
                output.append({
                    "rank": i,
                    "discord_id": row["user_id"],
                    "display_name": row["display_name"],
                    "xp": row["xp"],
                    "level": row["level"],
                    "challenges_completed": row["challenges_completed"],
                })
            return output
        finally:
            conn.close()


# ----------------------------
# Context manager and AI controller
# ----------------------------
class ContextManager:
    def __init__(self, store):
        self.store = store

    def build_context(self, user_id: str, message: str, channel_id: Optional[int] = None, is_private: bool = False) -> str:
        memories = self.store.get_recent_memories(user_id, limit=6)
        public_memories = self.store.get_public_memories(limit=5)
        progress = self.store.get_progress(user_id)

        context_parts = [
            f"User profile: XP={progress.get('xp', 0)}, Level={progress.get('level', 1)}, Challenges={progress.get('challenges_completed', 0)}",
        ]

        if memories:
            context_parts.append("Recent personal memories:\n- " + "\n- ".join(memories))

        if public_memories:
            context_parts.append("Recent public/community memories:\n- " + "\n- ".join(public_memories))

        if is_private:
            context_parts.append("Private context is allowed.")
        else:
            context_parts.append("Public channel context only. Do not reveal private memory.")

        return "\n\n".join(context_parts)


class ToolManager:
    def __init__(self, store: MovieDatabase):
        self.store = store

    def search_web(self, query: str) -> str:
        return f"Web search placeholder for: {query}"

    def search_python_news(self, topic: str = "python") -> str:
        return f"Python news placeholder for: {topic}"

    def get_user_profile(self, user_id: str) -> str:
        user = self.store.get_progress(user_id)
        return json.dumps(user, indent=2, default=str)

    def search_memory(self, user_id: str, term: str) -> str:
        mems = self.store.get_recent_memories(user_id, limit=10)
        hits = [m for m in mems if term.lower() in m.lower()]
        return ", ".join(hits) if hits else "No matching memory found."

    def create_memory(self, user_id: str, content: str, scope: str = "private") -> str:
        self.store.add_memory(user_id, content, scope=scope, memory_type="fact", importance="keep")
        return "Memory stored."

    def create_challenge(self, name: str, description: str, difficulty: str, xp: int = 100) -> str:
        self.store.add_challenge(name, description, difficulty, xp)
        return "Challenge created."

    def evaluate_submission(self, user_id: str, challenge_id: int, code: str) -> str:
        result = self.store.submit_challenge(user_id, challenge_id, code)
        if not result:
            return "Challenge not found."
        return f"Score: {result['score']}\nFeedback: {result['feedback']}"


class ChallengeSystem:
    def __init__(self, store: MovieDatabase):
        self.store = store
        self._seed_examples()

    def _seed_examples(self) -> None:
        if self.store.get_projects():
            return

        self.store.add_challenge(
            "List Comprehension Sprint",
            "Write a function that returns only even numbers from a list.",
            "easy",
            100,
        )

    def get_daily_challenge(self) -> Dict[str, Any]:
        challenges = [
            {
                "name": "List Comprehension Sprint",
                "description": "Write a function that returns only even numbers from a list.",
                "difficulty": "easy",
                "xp": 100,
                "hint": "Use a conditional inside the comprehension.",
                "solution": "return [n for n in numbers if n % 2 == 0]",
            },
            {
                "name": "Async Practice",
                "description": "Create an async function with await and return a value.",
                "difficulty": "intermediate",
                "xp": 200,
                "hint": "Use async def and await on a coroutine.",
                "solution": "async def example():\n    await asyncio.sleep(0.1)\n    return 42",
            },
        ]
        return challenges[0]

    def evaluate_submission(self, user_id: str, challenge: Dict[str, Any], code: str) -> Dict[str, Any]:
        score = 0
        if "return" in code:
            score += 40
        if "for" in code or "if" in code:
            score += 20
        if "async" in code or "await" in code:
            score += 15
        if len(code) > 20:
            score += 15
        if "def" in code:
            score += 10
        score = min(score, 100)

        feedback = "Good start. Tighten the logic and edge cases."
        if score >= 80:
            feedback = "Strong work. The logic is solid and the code is readable."
        elif score >= 50:
            feedback = "Reasonable attempt. The structure is there, but it still needs cleanup."

        return {"score": score, "feedback": feedback}


class AIController:
    def __init__(self, store: MovieDatabase):
        self.store = store
        self.context_manager = ContextManager(store)
        self.tools = ToolManager(store)
        self.challenges = ChallengeSystem(store)

    def classify_yes_no(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if "yes" in lowered:
            return True
        if "no" in lowered:
            return False
        return False

    def generate_response(self, user_id: str, user_name: str, channel_id: Optional[int], message: str, is_private: bool = False) -> str:
        context = self.context_manager.build_context(user_id, message, channel_id=channel_id, is_private=is_private)
        lower = message.lower()

        if "!pythonnews" in lower or "python news" in lower:
            return self.tools.search_python_news()

        if "!leaderboard" in lower:
            entries = self.store.leaderboard()
            if not entries:
                return "No leaderboard entries yet."
            rows = [f"{i.rank}. {i.display_name} — {i.xp} XP" for i in entries]
            return "\n".join(rows)

        if "!profile" in lower:
            progress = self.store.get_progress(user_id)
            return f"Profile for {user_name}:\n" + json.dumps(progress, indent=2)

        if "!challenge" in lower:
            challenge = self.challenges.get_daily_challenge()
            return f"Challenge: {challenge['name']}\n{challenge['description']}\nHint: {challenge['hint']}"

        if "!hint" in lower:
            return self.challenges.get_daily_challenge()["hint"]

        if "!solution" in lower:
            return self.challenges.get_daily_challenge()["solution"]

        if "project" in lower and "share" in lower:
            project_name = "Community Project"
            project_id = self.store.create_project(project_name, user_id)
            self.store.add_project_member(project_id, user_id, role="owner")
            return f"Project created: {project_name}. It is now shared and visible to the community."

        if "what did" in lower and "say" in lower and "project" in lower:
            public = self.store.get_public_memories(limit=5)
            if not public:
                return "I do not have a public project memory for that yet."
            return "Project memory:\n- " + "\n- ".join(public)

        self.store.add_memory(user_id, message, scope="public" if not is_private else "private", memory_type="fact", importance=score_memory_importance(message))

        return (
            f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser says: {message}\n"
            f"Reply as Movie in a calm, direct, technically precise tone. Keep it natural and useful."
        )


# ----------------------------
# Permission and command flow
# ----------------------------
@dataclass
class PermissionSet:
    user: bool = True
    moderator: bool = False
    admin: bool = False
    owner: bool = False

    def can(self, level: str) -> bool:
        levels = {"user": 1, "moderator": 2, "admin": 3, "owner": 4}
        current = levels.get(level, 1)
        score = 0
        if self.user:
            score = max(score, 1)
        if self.moderator:
            score = max(score, 2)
        if self.admin:
            score = max(score, 3)
        if self.owner:
            score = max(score, 4)
        return score >= current


class PermissionManager:
    def __init__(self):
        self.roles = dict()

    def get_permission(self, user_id: str) -> PermissionSet:
        return self.roles.get(user_id, PermissionSet())

    def set_permission(self, user_id: str, level: str) -> None:
        perms = self.get_permission(user_id)
        if level == "moderator":
            perms.moderator = True
        elif level == "admin":
            perms.admin = True
        elif level == "owner":
            perms.owner = True
        self.roles[user_id] = perms


# ----------------------------
# Discord bot
# ----------------------------
@dataclass
class MovieBotState:
    call_channels: Set[int] = field(default_factory=set)


class MovieBot:
    def __init__(self, token: str):
        self.token = token
        self.store = MovieDatabase()
        self.controller = AIController(self.store)
        self.permissions = PermissionManager()
        self.state = MovieBotState()
        self.bot = None

        if commands is not None and discord is not None:
            intents = discord.Intents.default()
            intents.message_content = True
            self.bot = commands.Bot(command_prefix="!", intents=intents)
            self._register_events()

    def _register_events(self) -> None:
        if self.bot is None:
            return

        @self.bot.event
        async def on_ready() -> None:
            print(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")

        @self.bot.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return

            raw = message.content.strip()
            if not raw:
                return

            self.store.upsert_user(str(message.author.id), str(message.author.display_name))
            self.store.resolve_pending_introductions(str(message.author.id), str(message.author.display_name))

            for name, intro_type in detect_introductions(raw):
                self.store.insert_introduction(
                    introduced_name=name,
                    introduced_by_id=str(message.author.id),
                    introduced_by_display_name=str(message.author.display_name),
                    intro_type=intro_type,
                    raw_text=raw,
                )

            if raw.startswith("!"):
                command = raw.split()[0].lower()
                if command == "!call":
                    self.state.call_channels.add(message.channel.id)
                    await message.channel.send("Movie is now active in this channel.")
                    return
                if command == "!stop":
                    self.state.call_channels.discard(message.channel.id)
                    await message.channel.send("Movie session closed for this channel.")
                    return
                if command == "!help":
                    await message.reply(
                        "Commands: !call, !stop, !help, !profile, !challenge, !hint, !solution, !leaderboard, !pythonnews, !project, !memory"
                    )
                    return
                if command == "!profile":
                    response = self.controller.generate_response(str(message.author.id), message.author.display_name, message.channel.id, "!profile", is_private=(message.guild is None))
                    await message.reply(response)
                    return
                if command == "!memory":
                    memories = self.store.get_recent_memories(str(message.author.id), 5)
                    await message.reply("Recent memory:\n- " + "\n- ".join(memories) if memories else "No stored memory yet.")
                    return
                if command == "!project":
                    project_id = self.store.create_project("Community Project", str(message.author.id))
                    self.store.add_project_member(project_id, str(message.author.id), role="owner")
                    await message.reply(f"Project created with ID {project_id}.")
                    return
                if command == "!pythonnews":
                    response = self.controller.generate_response(str(message.author.id), message.author.display_name, message.channel.id, "!pythonnews", is_private=(message.guild is None))
                    await message.reply(response)
                    return
                if command == "!leaderboard":
                    response = self.controller.generate_response(str(message.author.id), message.author.display_name, message.channel.id, "!leaderboard", is_private=(message.guild is None))
                    await message.reply(response)
                    return
                if command == "!challenge":
                    challenge = self.controller.challenges.get_daily_challenge()
                    await message.reply(f"{challenge['name']}\n{challenge['description']}\nHint: {challenge['hint']}")
                    return
                if command == "!hint":
                    await message.reply(self.controller.challenges.get_daily_challenge()["hint"])
                    return
                if command == "!solution":
                    await message.reply(self.controller.challenges.get_daily_challenge()["solution"])
                    return

            is_reply = message.reference is not None and getattr(message.reference, "resolved", None) is not None
            replied_to_bot = False
            if is_reply:
                ref = message.reference.resolved
                replied_to_bot = getattr(ref, "author", None) is not None and ref.author.bot

            addressed = bool(
                message.mention_everyone
                or any(user.id == self.bot.user.id for user in message.mentions)
                or replied_to_bot
                or (message.channel.id in self.state.call_channels)
                or name_trigger_score(raw, AI_NAME) >= 0.75
            )

            if not addressed:
                return

            response = self.controller.generate_response(
                str(message.author.id),
                message.author.display_name,
                message.channel.id,
                raw,
                is_private=(message.guild is None),
            )
            await message.reply(response)

    def run(self) -> None:
        if self.bot is None:
            raise RuntimeError("Discord dependencies are missing. Install discord.py to run the bot.")
        self.bot.run(self.token)


def _safe_env() -> str:
    load_dotenv()
    token = os.getenv("DISCORD_TOKEN") or os.getenv("BOT_TOKEN") or ""
    if not token:
        raise RuntimeError("Missing DISCORD_TOKEN or BOT_TOKEN environment variable.")
    return token


if __name__ == "__main__":
    try:
        token = _safe_env()
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)

    bot = MovieBot(token)
    bot.run()
