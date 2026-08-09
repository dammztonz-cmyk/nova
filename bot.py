#!/usr/bin/env python3
"""
Movie – AI Community Companion (single‑file, SQLite + optional PostgreSQL)
Implements:
- Natural name‑trigger with confidence scoring (Part D)
- Ambient introduction detection + retroactive linking (Part E)
- Memory importance: fast‑pass + LLM escalation (Part F)
- Discord: @mention, !call/!stop, reply‑to‑bot, voice transcription
- Gemini function calling (introduce_person, store_memory)
- SQLite (live) + PostgreSQL (permanent) two‑tier storage
"""

import os
import re
import json
import logging
import datetime
from typing import Optional, Dict, List, Any, Tuple

import discord
from discord.ext import commands
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- optional voice imports ----------
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ---------- optional PostgreSQL ----------
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# ---------- config ----------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "")
if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or GEMINI_API_KEY in .env")

AI_NAME = "Movie"
AI_ALIASES = ["movie", "movies"]
NAME_TRIGGER_AUTO_THRESHOLD = 0.75
NAME_TRIGGER_CONFIRM_FLOOR = 0.35
SQLITE_DB_PATH = "movie_bot.db"
USE_POSTGRES = bool(POSTGRES_DSN) and POSTGRES_AVAILABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MovieBot")

# ---------- Gemini ----------
genai.configure(api_key=GEMINI_API_KEY)

# ===================== DATABASE ABSTRACTION LAYER =====================
# Unified interface for SQLite and PostgreSQL

def get_db_connection():
    """Return (connection, cursor) for the active database."""
    if USE_POSTGRES:
        conn = psycopg2.connect(POSTGRES_DSN)
        return conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn, conn.cursor()

def execute_query(query: str, params: tuple = (), fetch_one=False, fetch_all=False):
    conn, cur = get_db_connection()
    cur.execute(query, params)
    result = None
    if fetch_one:
        result = cur.fetchone()
    elif fetch_all:
        result = cur.fetchall()
    conn.commit()
    conn.close()
    return result

def init_db():
    if USE_POSTGRES:
        conn, cur = get_db_connection()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                discord_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS introductions (
                id SERIAL PRIMARY KEY,
                introduced_name TEXT NOT NULL,
                introduced_by_id TEXT,
                introduced_by_display_name TEXT NOT NULL,
                intro_type TEXT NOT NULL,
                raw_text TEXT,
                linked_discord_id TEXT REFERENCES users(discord_id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                channel_id TEXT,
                guild_id TEXT,
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                ended_at TIMESTAMPTZ,
                active BOOLEAN DEFAULT TRUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES conversations(id),
                role TEXT,
                content TEXT,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(discord_id),
                type TEXT,
                content TEXT,
                importance INTEGER DEFAULT 1,
                visibility TEXT DEFAULT 'private',
                source_conversation_id INTEGER REFERENCES conversations(id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                metadata JSONB
            )
        """)
        conn.commit()
        conn.close()
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                discord_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                first_seen TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
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
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                channel_id TEXT,
                guild_id TEXT,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                active BOOLEAN DEFAULT 1
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                type TEXT,
                content TEXT,
                importance INTEGER DEFAULT 1,
                visibility TEXT DEFAULT 'private',
                source_conversation_id INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(discord_id),
                FOREIGN KEY (source_conversation_id) REFERENCES conversations(id)
            )
        """)
        conn.commit()
        conn.close()
init_db()

# ---------- database helpers ----------
def upsert_user(discord_id: str, display_name: str):
    """Insert or update user, and retroactively link pending introductions."""
    if USE_POSTGRES:
        execute_query(
            "INSERT INTO users (discord_id, display_name) VALUES (%s, %s) ON CONFLICT (discord_id) DO UPDATE SET display_name = EXCLUDED.display_name",
            (discord_id, display_name)
        )
        rows = execute_query(
            "SELECT id FROM introductions WHERE linked_discord_id IS NULL AND introduced_name = %s",
            (display_name,), fetch_all=True
        )
        if rows:
            for row in rows:
                execute_query("UPDATE introductions SET linked_discord_id = %s WHERE id = %s", (discord_id, row["id"]))
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT discord_id FROM users WHERE discord_id = ?", (discord_id,))
        if c.fetchone() is None:
            c.execute("INSERT INTO users (discord_id, display_name) VALUES (?, ?)", (discord_id, display_name))
            c.execute("SELECT id FROM introductions WHERE linked_discord_id IS NULL AND introduced_name = ?", (display_name,))
            pending = c.fetchall()
            for row in pending:
                c.execute("UPDATE introductions SET linked_discord_id = ? WHERE id = ?", (discord_id, row[0]))
        else:
            c.execute("UPDATE users SET display_name = ? WHERE discord_id = ?", (display_name, discord_id))
        conn.commit()
        conn.close()

def get_or_create_conversation(user_id: str, channel_id: str, guild_id: Optional[str] = None) -> int:
    if USE_POSTGRES:
        row = execute_query(
            "SELECT id FROM conversations WHERE user_id = %s AND channel_id = %s AND active = true",
            (user_id, channel_id), fetch_one=True
        )
        if row:
            return row["id"]
        execute_query(
            "INSERT INTO conversations (user_id, channel_id, guild_id) VALUES (%s, %s, %s)",
            (user_id, channel_id, guild_id)
        )
        row = execute_query("SELECT lastval()", fetch_one=True)
        return row["lastval"]
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM conversations WHERE user_id = ? AND channel_id = ? AND active = 1", (user_id, channel_id))
        row = c.fetchone()
        if row:
            conn.close()
            return row[0]
        c.execute("INSERT INTO conversations (user_id, channel_id, guild_id) VALUES (?, ?, ?)", (user_id, channel_id, guild_id))
        conn.commit()
        c.execute("SELECT last_insert_rowid()")
        conv_id = c.fetchone()[0]
        conn.close()
        return conv_id

def add_message(conversation_id: int, role: str, content: str):
    if USE_POSTGRES:
        execute_query(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content)
        )
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (conversation_id, role, content))
        conn.commit()
        conn.close()

def get_recent_messages(conversation_id: int, limit: int = 20) -> List[Dict]:
    if USE_POSTGRES:
        rows = execute_query(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY timestamp DESC LIMIT %s",
            (conversation_id, limit), fetch_all=True
        )
        return [dict(row) for row in reversed(rows)] if rows else []
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?", (conversation_id, limit))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in reversed(rows)]

def store_memory(user_id: Optional[str], mem_type: str, content: str,
                 importance: int = 1, visibility: str = "private",
                 source_conv: Optional[int] = None, metadata: Optional[Dict] = None):
    meta_json = json.dumps(metadata) if metadata else None
    if USE_POSTGRES:
        execute_query(
            """INSERT INTO memories (user_id, type, content, importance, visibility,
               source_conversation_id, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_id, mem_type, content, importance, visibility, source_conv, meta_json)
        )
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT INTO memories (user_id, type, content, importance, visibility,
               source_conversation_id, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, mem_type, content, importance, visibility, source_conv, meta_json)
        )
        conn.commit()
        conn.close()

def get_memories(user_id: Optional[str] = None, visibility: Optional[str] = None,
                 mem_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
    if USE_POSTGRES:
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        if visibility:
            query += " AND visibility = %s"
            params.append(visibility)
        if mem_type:
            query += " AND type = %s"
            params.append(mem_type)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        rows = execute_query(query, tuple(params), fetch_all=True)
        return [dict(row) for row in rows] if rows else []
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if visibility:
            query += " AND visibility = ?"
            params.append(visibility)
        if mem_type:
            query += " AND type = ?"
            params.append(mem_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

# ===================== IDENTITY: NAME TRIGGER (Part D) =====================
class NameTrigger:
    @staticmethod
    def evaluate(text: str) -> float:
        """Return confidence (0.0-1.0) that the message is addressing the AI by name."""
        text_lower = text.lower()
        confidence = 0.0
        if any(alias in text_lower for alias in AI_ALIASES):
            confidence += 0.3
        if re.search(rf"^(?:hey|yo|ok|okay|so|well|hmm)?\s*{AI_NAME}\b", text_lower):
            confidence += 0.45
        if re.search(rf"{AI_NAME}\s*[?!]\s*$", text_lower):
            confidence += 0.35
        if '?' in text:
            confidence += 0.10
        if len(text) <= len(AI_NAME) + 6:
            confidence += 0.3
        if re.search(rf"(?:a |the |this ){AI_NAME}\b", text_lower):
            confidence -= 0.55
        if re.search(rf"(?:watch|watching|watched|see|rent|stream)\s+{AI_NAME}\b", text_lower):
            confidence -= 0.55
        return max(0.0, min(1.0, confidence))

# ===================== IDENTITY: INTRODUCTIONS (Part E) =====================
class IntroductionDetector:
    THIRD_PARTY_PATTERNS = [
        r"(?i:this is )([A-Z][a-z]+)",
        r"(?i:meet )([A-Z][a-z]+)",
        r"(?i:([A-Z][a-z]+) is (?:joining|new|just joined))",
        r"(?i:(?:I'd like you all|everyone) to meet )([A-Z][a-z]+)",
        r"(?i:introduc(?:e|ing) )([A-Z][a-z]+)",
    ]
    SELF_PATTERNS = [
        r"(?i:i'm )([A-Z][a-z]+)",
        r"(?i:my name is )([A-Z][a-z]+)",
        r"(?i:call me )([A-Z][a-z]+)",
    ]
    STOPWORDS = {"good", "fine", "sorry", "sure", "ok", "okay", "back", "here", "done", "not", "just", "still", "also", "new", "trying"}

    @classmethod
    def detect(cls, text: str, author_display_name: str) -> List[Dict]:
        results = []
        for pattern in cls.THIRD_PARTY_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                if name.lower() not in cls.STOPWORDS:
                    results.append({"name": name, "type": "third_party", "raw": text})
                    break
        for pattern in cls.SELF_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                if name.lower() not in cls.STOPWORDS and name.lower() == author_display_name.lower():
                    results.append({"name": name, "type": "self", "raw": text})
                    break
        return results

# ===================== MEMORY IMPORTANCE (Part F) =====================
class ImportanceScorer:
    @staticmethod
    def fast_scan(text: str) -> Tuple[str, int]:
        """Return (decision, importance) where decision is 'discard', 'keep', or 'ambiguous'."""
        keep_patterns = [
            r"(?:i'm|i am) (?:building|working on|creating|making) (?:a|an|the)?\s*(?:discord|bot|game|app|project)",
            r"(?:my|our) (?:project|goal|plan) is",
            r"(?:allergy|allergic to|pronouns|location|from|live in)",
            r"(?:prefer|preference|favorite)",
            r"(?:decided|decision|agree|agreed)",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ]
        for pat in keep_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return ("keep", 3)
        discard_patterns = [
            r"(?:i'm|i am) (?:eating|drinking|going to|sleeping|tired|hungry)",
            r"(?:weather|today|tomorrow|yesterday)",
        ]
        for pat in discard_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return ("discard", 0)
        if re.search(r"\b(?:i|we|my|our)\s+\w+\b", text):
            return ("ambiguous", 1)
        return ("discard", 0)

    @staticmethod
    def llm_escalate(text_window: str) -> Tuple[bool, int]:
        prompt = f"""Analyze the following recent conversation messages (separated by newlines) and decide if any part of it contains a fact, preference, project, or personal detail that should be remembered permanently.
        If yes, output a single line with: YES,<importance> where importance is 1-5 (1=low, 5=critical).
        If no, output: NO,0.
        Messages:
        {text_window}
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if text.startswith("YES"):
                parts = text.split(",")
                if len(parts) > 1:
                    imp = int(parts[1].strip())
                    return (True, min(5, max(1, imp)))
            return (False, 0)
        except Exception:
            return (False, 0)

# ===================== AI CONTROLLER & GEMINI CLIENT =====================
class GeminiClient:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": 0.7, "max_output_tokens": 1024},
            tools=self._get_tools()
        )

    def _get_tools(self):
        return [
            {
                "name": "introduce_person",
                "description": "Record that a person has been introduced.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "introduced_by": {"type": "string"},
                        "visibility": {"type": "string", "enum": ["private","shared","public"]}
                    },
                    "required": ["name","introduced_by"]
                }
            },
            {
                "name": "store_memory",
                "description": "Store a fact, preference, project, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "type": {"type": "string", "enum": ["fact","preference","project","person","other"]},
                        "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                        "visibility": {"type": "string", "enum": ["private","shared","public"]}
                    },
                    "required": ["content","type"]
                }
            }
        ]

    def generate_response(self, prompt_text: str, conv_id: int, user_id: str) -> str:
        response = self.model.generate_content(prompt_text)
        function_called = False
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    self._handle_function_call(fc.name, fc.args, conv_id, user_id)
                    function_called = True
            text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
            final_text = "".join(text_parts) if text_parts else ""
            if not final_text and function_called:
                # fallback acknowledgment
                if fc and fc.name == "introduce_person":
                    name = fc.args.get("name", "someone")
                    final_text = f"Noted, {name}."
                else:
                    final_text = "Memory stored."
        else:
            final_text = response.text if response.text else "I didn't understand that."

        add_message(conv_id, "assistant", final_text)
        return final_text

    def _handle_function_call(self, name: str, args: Dict, conv_id: int, user_id: str):
        if name == "introduce_person":
            person = args.get("name")
            introduced_by = args.get("introduced_by", "unknown")
            visibility = args.get("visibility", "private")
            if person:
                store_memory(
                    user_id=user_id,
                    mem_type="person",
                    content=f"Person introduced: {person}",
                    importance=3,
                    visibility=visibility,
                    source_conv=conv_id,
                    metadata={"name": person, "introduced_by": introduced_by}
                )
                # also store in introductions table for retroactive linking
                if USE_POSTGRES:
                    execute_query(
                        """INSERT INTO introductions (introduced_name, introduced_by_id, introduced_by_display_name,
                           intro_type, raw_text, linked_discord_id) VALUES (%s, %s, %s, %s, %s, %s)""",
                        (person, user_id, introduced_by, "third_party", f"introduced by {introduced_by}", None)
                    )
                else:
                    import sqlite3
                    conn = sqlite3.connect(SQLITE_DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        """INSERT INTO introductions (introduced_name, introduced_by_id, introduced_by_display_name,
                           intro_type, raw_text, linked_discord_id) VALUES (?, ?, ?, ?, ?, ?)""",
                        (person, user_id, introduced_by, "third_party", f"introduced by {introduced_by}", None)
                    )
                    conn.commit()
                    conn.close()
                logger.info(f"Introduction stored: {person} by {introduced_by}")
        elif name == "store_memory":
            content = args.get("content")
            mem_type = args.get("type", "fact")
            importance = args.get("importance", 2)
            visibility = args.get("visibility", "private")
            if content:
                store_memory(user_id, mem_type, content, importance, visibility, conv_id)
                logger.info(f"Memory stored: {content}")

# ===================== DISCORD BOT =====================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)
gemini_client = GeminiClient()
call_sessions = {}  # (user_id, channel_id) -> bool

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")
    await bot.change_presence(activity=discord.Game(name=f"Python | @{AI_NAME}"))

# ---------- system prompt ----------
def get_system_prompt() -> str:
    return f"""You are nova, a calm, intelligent, and natural AI companion living in a Discord server.
You are an expert in Python programming and enjoy teaching, mentoring, and collaborating.
Your personality: conversational, technically precise, occasionally humorous, never over‑enthusiastic.
You are willing to say "I don't know" and to correct users when needed.
You can switch between teaching mode and casual chat effortlessly.

Important instructions:
- You have the ability to store memories. When a user says something important about themselves, a project, or another person, you should store it.
- **Natural introductions**: If a user says something like "This is John", "Meet Sarah", "Introducing Alex", or any phrase that clearly introduces a person, you MUST call the `introduce_person` function. Provide the person's name, the introducer (the user who said it), and set visibility to "private" initially (the introducer can later share it).
- You can also call `store_memory` for other facts (e.g., "I'm building a game" → store as 'project' or 'fact').
- When responding, be concise and natural. Use markdown for code blocks if needed.
- If you're unsure about something, say so.
- Avoid robotic phrases like "That's a great question!" every time. Vary your responses. u were build by x just x no company no existence"""

# ---------- ambient processing (introductions + memory) ----------
async def process_ambient(message: discord.Message):
    """Run on every message Movie can see, regardless of response."""
    user = message.author
    # 1) introductions (Part E)
    intro_candidates = IntroductionDetector.detect(message.content, user.display_name)
    for intro in intro_candidates:
        # check if user already exists with that display name
        linked_id = None
        if USE_POSTGRES:
            row = execute_query("SELECT discord_id FROM users WHERE display_name = %s", (intro["name"],), fetch_one=True)
            if row:
                linked_id = row["discord_id"]
            execute_query(
                """INSERT INTO introductions (introduced_name, introduced_by_id, introduced_by_display_name,
                   intro_type, raw_text, linked_discord_id) VALUES (%s, %s, %s, %s, %s, %s)""",
                (intro["name"], str(user.id), user.display_name, intro["type"], intro["raw"], linked_id)
            )
        else:
            import sqlite3
            conn = sqlite3.connect(SQLITE_DB_PATH)
            c = conn.cursor()
            c.execute("SELECT discord_id FROM users WHERE display_name = ?", (intro["name"],))
            row = c.fetchone()
            linked_id = row[0] if row else None
            c.execute(
                """INSERT INTO introductions (introduced_name, introduced_by_id, introduced_by_display_name,
                   intro_type, raw_text, linked_discord_id) VALUES (?, ?, ?, ?, ?, ?)""",
                (intro["name"], str(user.id), user.display_name, intro["type"], intro["raw"], linked_id)
            )
            conn.commit()
            conn.close()
        logger.info(f"Ambient intro: {intro['name']} ({intro['type']})")

    # 2) memory importance (Part F)
    decision, importance = ImportanceScorer.fast_scan(message.content)
    if decision == "keep":
        # store directly in memories (SQLite) – later we could promote to Postgres
        store_memory(str(user.id), "fact", message.content, importance, "private")
        logger.info(f"Memory kept (fast): {message.content[:50]}...")
    elif decision == "ambiguous":
        # get surrounding messages for context (last 5)
        conv_id = get_or_create_conversation(str(user.id), str(message.channel.id), str(message.guild.id) if message.guild else None)
        recent = get_recent_messages(conv_id, limit=5)
        window = "\n".join([f"{m['role']}: {m['content']}" for m in recent] + [f"user: {message.content}"])
        keep, imp = ImportanceScorer.llm_escalate(window)
        if keep:
            store_memory(str(user.id), "fact", message.content, imp, "private")
            logger.info(f"Memory kept (LLM): {message.content[:50]}...")

# ---------- main message handler ----------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # handle commands
    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    # determine if we should respond
    should_respond = False
    if bot.user in message.mentions:
        should_respond = True
    if call_sessions.get((message.author.id, message.channel.id), False):
        should_respond = True
    if (message.reference and message.reference.resolved and
        message.reference.resolved.author.id == bot.user.id):
        should_respond = True
    if not should_respond:
        confidence = NameTrigger.evaluate(message.content)
        if confidence >= NAME_TRIGGER_AUTO_THRESHOLD:
            should_respond = True
        elif confidence >= NAME_TRIGGER_CONFIRM_FLOOR:
            # escalate to Gemini Flash for cheap yes/no
            classification_prompt = f"Is this message directly addressing a participant named '{AI_NAME}'? Answer only 'yes' or 'no'.\nMessage: {message.content}"
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(classification_prompt)
            if resp.text and "yes" in resp.text.lower():
                should_respond = True

    # always process ambient (introductions, memory) before potentially responding
    await process_ambient(message)

    if not should_respond:
        return

    # ---------- we are responding ----------
    user = message.author
    upsert_user(str(user.id), user.display_name)

    # voice transcription
    voice_transcript = None
    if message.attachments:
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("audio/"):
                voice_transcript = await transcribe_attachment(att)
                if voice_transcript:
                    message.content = f"[Voice: {voice_transcript}] " + message.content
                break

    # conversation
    guild_id = str(message.guild.id) if message.guild else None
    conv_id = get_or_create_conversation(str(user.id), str(message.channel.id), guild_id)
    add_message(conv_id, "user", message.content)

    # build context
    history = get_recent_messages(conv_id, limit=20)
    memories = get_memories(user_id=str(user.id), visibility="private", limit=5)
    prompt = f"### System Instructions ###\n{get_system_prompt()}\n\n"
    if message.reference and message.reference.resolved:
        prompt += f"### Replying to: {message.reference.resolved.content}\n\n"
    for mem in memories:
        prompt += f"- Memory: {mem['content']} (type: {mem['type']}, importance: {mem['importance']})\n"
    if history:
        prompt += "\n### Conversation history ###\n"
        for msg in history:
            prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"\n### Current message from user ###\n{message.content}\n\n### Your response: ###\n"

    # get AI response
    try:
        response = gemini_client.generate_response(prompt, conv_id, str(user.id))
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        response = "Sorry, I had a problem. Please try again."

    await message.reply(response, mention_author=False)

# ---------- voice transcription helper ----------
async def transcribe_attachment(attachment):
    if not WHISPER_AVAILABLE:
        logger.warning("Whisper not installed; voice transcription disabled.")
        return None
    try:
        temp_path = f"temp_audio_{attachment.id}.{attachment.filename.split('.')[-1]}"
        await attachment.save(temp_path)
        model = whisper.load_model("base")
        result = model.transcribe(temp_path)
        os.remove(temp_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        return None

# ---------- commands ----------
@bot.command(name="call")
async def call_mode(ctx):
    call_sessions[(ctx.author.id, ctx.channel.id)] = True
    await ctx.send(f"Call mode activated, {ctx.author.mention}. I'm listening. Use `!stop` to end.")

@bot.command(name="stop")
async def stop_mode(ctx):
    if call_sessions.pop((ctx.author.id, ctx.channel.id), None):
        await ctx.send(f"Call mode ended, {ctx.author.mention}.")
    else:
        await ctx.send("You weren't in a call session.")

# ---------- run ----------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("No DISCORD_TOKEN found.")
    else:
        bot.run(DISCORD_TOKEN)
