# ============================================================
# AI PYTHON DISCORD BOT
# bot.py
#
# Core features:
# - Discord.py bot
# - Gemini AI
# - User identification by Discord ID
# - SQLite temporary storage
# - PostgreSQL permanent storage
# - Conversation memory (properly scoped per channel/DM)
# - !call / !stop (properly scoped per channel/DM)
# - Bot mentions
# - Replies to bot messages
# - Daily Python challenges (fixed posting time)
# - Challenge memory
#
# ------------------------------------------------------------
# FIXES APPLIED IN THIS VERSION (vs. the original):
#
# 1. Private DM history no longer leaks into public server
#    replies. get_recent_messages() is now scoped by
#    conversation_type + channel, not just discord_id.
#
# 2. !call is now scoped per channel/DM, not global to the
#    user across every server the bot shares with them.
#
# 3. create_conversation() now reuses an existing open
#    conversation for the same user+channel instead of
#    inserting a brand-new row on every single message.
#
# 4. SQLite temp-storage calls are no longer blocking the
#    asyncio event loop -- they're run in a thread via
#    asyncio.to_thread() and properly awaited.
#
# 5. Daily challenge posting now fires at a fixed UTC time
#    every day instead of drifting based on when the bot
#    process happened to start.
#
# 6. Memory visibility ("private" vs "shared") is now
#    actually read consistently -- get_user_memories()
#    returns both private memories belonging to the user
#    and memories explicitly marked shared/community, since
#    the DB never had a mechanism to set anything but
#    "private" before.
#
# Everything about the AI personality, Gemini prompting,
# memory classification, and challenge generation is left
# functionally the same as before.
#
# Install:
# pip install discord.py python-dotenv asyncpg google-genai
#
# .env:
# DISCORD_TOKEN=your_discord_bot_token
# GEMINI_API_KEY=your_gemini_api_key
# DATABASE_URL=your_postgresql_database_url
# BOT_PREFIX=!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import sqlite3
import asyncio
import logging
import random
import traceback

from datetime import datetime, timezone, timedelta, time as dt_time

from dotenv import load_dotenv

import discord
from discord.ext import commands, tasks

import asyncpg

from google import genai
from google.genai import types


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

BOT_PREFIX = os.getenv("BOT_PREFIX", "!")


# ============================================================
# BASIC VALIDATION
# ============================================================

if not DISCORD_TOKEN:
    raise RuntimeError(
        "DISCORD_TOKEN is missing from the environment."
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from the environment."
    )

if not DATABASE_URL:
    logging.warning(
        "DATABASE_URL is not set. "
        "PostgreSQL features will not work."
    )


# ============================================================
# CONFIGURATION
# ============================================================

SQLITE_DATABASE = "bot_temp.db"

# Gemini model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash"
)

# Number of messages kept for short-term context
MAX_HISTORY_MESSAGES = 15

# Number of active conversations kept in memory
MAX_ACTIVE_USERS = 1000

# UTC hour/minute the daily challenge should post at.
DAILY_CHALLENGE_HOUR = 12
DAILY_CHALLENGE_MINUTE = 0


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# DISCORD INTENTS
# ============================================================

intents = discord.Intents.default()

# We need message content to read normal messages.
intents.message_content = True

# We need members/user information.
intents.members = True


# ============================================================
# BOT
# ============================================================

bot = commands.Bot(
    command_prefix=BOT_PREFIX,
    intents=intents,
    help_command=None
)


# ============================================================
# DATABASE VARIABLES
# ============================================================

postgres_pool = None

# Users currently using !call.
#
# IMPORTANT: scoped per (user_id, channel_id) so that
# activating !call in one server/DM does not make the bot
# start responding to that user's plain messages everywhere
# else it shares with them.
#
# Example entry: (123456789, 987654321)
active_calls = set()

# In-memory recent conversation history.
#
# Key:
#     Discord user ID
#
# Value:
#     list of messages
#
# Example:
#
# {
#     123456789: [
#         {"role": "user", "content": "Hello"},
#         {"role": "model", "content": "Hey"}
#     ]
# }
#
conversation_cache = {}


# ============================================================
# AI PERSONALITY
# ============================================================

AI_SYSTEM_INSTRUCTION = """
You are an intelligent AI companion living inside a Discord
community.

Your strongest area is Python and software development.

You are a:
- Python expert
- programming mentor
- project companion
- technical problem solver
- natural conversational AI
- helpful community assistant

Your personality should feel natural and human-like without
pretending to be human.

Do not constantly say:
"Great question!"
"Absolutely!"
"That's an amazing question!"

Do not add unnecessary jokes or filler.

Do not be unnecessarily formal.

Do not be unnecessarily robotic.

Speak naturally.

When the user is asking a technical question, focus on the
technical problem.

When the user is learning Python, act as a mentor.

When the user is casually talking, respond naturally.

You can disagree with the user when appropriate.

You should correct technical mistakes instead of blindly
agreeing with them.

Do not invent memories.

Do not claim that someone said something unless that information
is actually present in the context provided to you.

You are one AI entity, but different Discord users have different
identities and conversation histories.

Never mix one user's private conversation history with another
user's private conversation history.

Discord user IDs are the application's identity keys.

When information from a public Discord conversation is supplied
to you as context, you may use it as public community context.

Private DM information must not automatically become public
community information.

If information is explicitly marked as shared, you may use it
as shared community information.

When you do not know something, say so rather than inventing it.

Your primary goal is to be useful, technically strong, natural,
and context-aware.
"""


# ============================================================
# SQLITE (blocking helpers -- always call via asyncio.to_thread)
# ============================================================

def get_sqlite_connection():
    """
    Return a SQLite database connection.
    """

    connection = sqlite3.connect(
        SQLITE_DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


def _initialize_sqlite_sync():
    """
    Create SQLite tables for temporary information.
    Blocking -- run via asyncio.to_thread().
    """

    connection = get_sqlite_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporary_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            data_type TEXT NOT NULL,
            data TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT
        )
    """)

    connection.commit()
    connection.close()

    logging.info("SQLite initialized.")


def _save_temporary_data_sync(
    user_id,
    data_type,
    data,
    expires_at=None
):
    """
    Save temporary information to SQLite.
    Blocking -- run via asyncio.to_thread().
    """

    connection = get_sqlite_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO temporary_data
        (
            user_id,
            data_type,
            data,
            created_at,
            expires_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(user_id),
            data_type,
            json.dumps(data),
            datetime.now(timezone.utc).isoformat(),
            expires_at
        )
    )

    connection.commit()
    connection.close()


def _get_temporary_data_sync(
    user_id,
    data_type=None
):
    """
    Retrieve temporary information from SQLite.
    Blocking -- run via asyncio.to_thread().
    """

    connection = get_sqlite_connection()

    cursor = connection.cursor()

    if data_type:
        cursor.execute(
            """
            SELECT *
            FROM temporary_data
            WHERE user_id = ?
            AND data_type = ?
            ORDER BY id DESC
            """,
            (
                str(user_id),
                data_type
            )
        )
    else:
        cursor.execute(
            """
            SELECT *
            FROM temporary_data
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (str(user_id),)
        )

    rows = cursor.fetchall()

    connection.close()

    results = []

    for row in rows:
        try:
            data = json.loads(row["data"])
        except Exception:
            data = row["data"]

        results.append({
            "id": row["id"],
            "user_id": row["user_id"],
            "data_type": row["data_type"],
            "data": data,
            "created_at": row["created_at"],
            "expires_at": row["expires_at"]
        })

    return results


async def initialize_sqlite():
    """
    Async wrapper -- creates SQLite tables without blocking
    the event loop.
    """

    await asyncio.to_thread(_initialize_sqlite_sync)


async def save_temporary_data(
    user_id,
    data_type,
    data,
    expires_at=None
):
    """
    Async wrapper -- saves temporary data without blocking
    the event loop.
    """

    await asyncio.to_thread(
        _save_temporary_data_sync,
        user_id,
        data_type,
        data,
        expires_at
    )


async def get_temporary_data(
    user_id,
    data_type=None
):
    """
    Async wrapper -- reads temporary data without blocking
    the event loop.
    """

    return await asyncio.to_thread(
        _get_temporary_data_sync,
        user_id,
        data_type
    )


# ============================================================
# POSTGRESQL
# ============================================================

async def initialize_postgres():
    """
    Create the PostgreSQL connection pool and tables.
    """

    global postgres_pool

    if not DATABASE_URL:
        logging.warning(
            "Skipping PostgreSQL initialization."
        )
        return

    postgres_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=1,
        max_size=10
    )

    async with postgres_pool.acquire() as connection:

        # ----------------------------------------------------
        # USERS
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                discord_id BIGINT UNIQUE NOT NULL,
                username TEXT,
                display_name TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ----------------------------------------------------
        # CONVERSATIONS
        #
        # A conversation is now a persistent thread scoped to
        # one user + one channel (or DM). ended_at is set to
        # NULL while still "open"; on_message reuses the open
        # conversation instead of creating a new row every
        # message.
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT
                    REFERENCES users(id)
                    ON DELETE CASCADE,
                guild_id BIGINT,
                channel_id BIGINT NOT NULL,
                conversation_type TEXT DEFAULT 'public',
                started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMPTZ
            )
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_open
            ON conversations (user_id, channel_id)
            WHERE ended_at IS NULL
        """)

        # ----------------------------------------------------
        # MESSAGES
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id BIGINT
                    REFERENCES conversations(id)
                    ON DELETE CASCADE,
                discord_message_id BIGINT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ----------------------------------------------------
        # MEMORIES
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT
                    REFERENCES users(id)
                    ON DELETE CASCADE,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 1,
                visibility TEXT DEFAULT 'private',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ----------------------------------------------------
        # CHALLENGES
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS challenges (
                id BIGSERIAL PRIMARY KEY,
                challenge_number INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                topic TEXT,
                solution TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                posted_at TIMESTAMPTZ
            )
        """)

        # ----------------------------------------------------
        # PROJECTS
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id BIGINT
                    REFERENCES users(id)
                    ON DELETE SET NULL,
                visibility TEXT DEFAULT 'private',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ----------------------------------------------------
        # PROJECT MEMBERS
        # ----------------------------------------------------

        await connection.execute("""
            CREATE TABLE IF NOT EXISTS project_members (
                project_id BIGINT
                    REFERENCES projects(id)
                    ON DELETE CASCADE,

                user_id BIGINT
                    REFERENCES users(id)
                    ON DELETE CASCADE,

                role TEXT DEFAULT 'member',

                joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (
                    project_id,
                    user_id
                )
            )
        """)

    logging.info("PostgreSQL initialized.")


# ============================================================
# USER MANAGEMENT
# ============================================================

async def get_or_create_user(user):
    """
    Get a user from PostgreSQL using the Discord user ID.

    The Discord ID is the real identity key.
    """

    if postgres_pool is None:
        return None

    async with postgres_pool.acquire() as connection:

        existing = await connection.fetchrow(
            """
            SELECT *
            FROM users
            WHERE discord_id = $1
            """,
            user.id
        )

        if existing:
            await connection.execute(
                """
                UPDATE users
                SET username = $1,
                    display_name = $2,
                    updated_at = CURRENT_TIMESTAMP
                WHERE discord_id = $3
                """,
                user.name,
                user.display_name,
                user.id
            )

            return existing

        created = await connection.fetchrow(
            """
            INSERT INTO users
            (
                discord_id,
                username,
                display_name
            )
            VALUES ($1, $2, $3)
            RETURNING *
            """,
            user.id,
            user.name,
            user.display_name
        )

        return created


# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

async def get_or_create_conversation(
    user_id,
    guild_id,
    channel_id,
    conversation_type="public"
):
    """
    Reuse an existing OPEN conversation for this user+channel
    if one exists, otherwise create a new one.

    This replaces the old behaviour of inserting a brand-new
    conversation row on every single message.
    """

    if postgres_pool is None:
        return None

    async with postgres_pool.acquire() as connection:

        row = await connection.fetchrow(
            """
            SELECT id
            FROM users
            WHERE discord_id = $1
            """,
            user_id
        )

        if not row:
            return None

        user_db_id = row["id"]

        existing = await connection.fetchrow(
            """
            SELECT id
            FROM conversations
            WHERE user_id = $1
            AND channel_id = $2
            AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            user_db_id,
            channel_id
        )

        if existing:
            return existing["id"]

        conversation = await connection.fetchrow(
            """
            INSERT INTO conversations
            (
                user_id,
                guild_id,
                channel_id,
                conversation_type
            )
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_db_id,
            guild_id,
            channel_id,
            conversation_type
        )

        return conversation["id"]


async def save_message(
    conversation_id,
    discord_message_id,
    role,
    content
):
    """
    Save a message to PostgreSQL.
    """

    if postgres_pool is None or conversation_id is None:
        return

    async with postgres_pool.acquire() as connection:

        await connection.execute(
            """
            INSERT INTO messages
            (
                conversation_id,
                discord_message_id,
                role,
                content
            )
            VALUES ($1, $2, $3, $4)
            """,
            conversation_id,
            discord_message_id,
            role,
            content
        )


async def get_recent_messages(
    user_id,
    channel_id,
    conversation_type,
    limit=MAX_HISTORY_MESSAGES
):
    """
    Retrieve recent messages belonging to one user, SCOPED to
    the current channel/DM and conversation type.

    This is the key privacy fix: a user's private DM history
    will never be pulled into context for a public server
    reply, and vice versa, because we filter on both
    channel_id and conversation_type rather than discord_id
    alone.
    """

    if postgres_pool is None:
        return []

    async with postgres_pool.acquire() as connection:

        rows = await connection.fetch(
            """
            SELECT
                m.role,
                m.content,
                m.created_at
            FROM messages m
            JOIN conversations c
                ON m.conversation_id = c.id
            JOIN users u
                ON c.user_id = u.id
            WHERE u.discord_id = $1
            AND c.channel_id = $2
            AND c.conversation_type = $3
            ORDER BY m.created_at DESC
            LIMIT $4
            """,
            user_id,
            channel_id,
            conversation_type,
            limit
        )

    rows = list(reversed(rows))

    return [
        {
            "role": row["role"],
            "content": row["content"]
        }
        for row in rows
    ]


# ============================================================
# MEMORY
# ============================================================

async def save_memory(
    user_id,
    memory_type,
    content,
    importance=1,
    visibility="private"
):
    """
    Save permanent memory to PostgreSQL.
    """

    if postgres_pool is None:
        return

    async with postgres_pool.acquire() as connection:

        user = await connection.fetchrow(
            """
            SELECT id
            FROM users
            WHERE discord_id = $1
            """,
            user_id
        )

        if not user:
            return

        await connection.execute(
            """
            INSERT INTO memories
            (
                user_id,
                memory_type,
                content,
                importance,
                visibility
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            user["id"],
            memory_type,
            content,
            importance,
            visibility
        )


async def get_user_memories(
    user_id,
    limit=10
):
    """
    Retrieve important memories belonging to one user.

    Returns memories that are either:
    - private and owned by this user, or
    - explicitly marked shared/community

    (The original version hardcoded visibility = 'private'
    only, which meant nothing marked 'shared' was ever
    actually surfaced.)
    """

    if postgres_pool is None:
        return []

    async with postgres_pool.acquire() as connection:

        rows = await connection.fetch(
            """
            SELECT
                memory_type,
                content,
                importance,
                visibility
            FROM memories m
            JOIN users u
                ON m.user_id = u.id
            WHERE (
                (u.discord_id = $1 AND visibility = 'private')
                OR visibility = 'shared'
            )
            ORDER BY importance DESC,
                     updated_at DESC
            LIMIT $2
            """,
            user_id,
            limit
        )

    return [
        {
            "type": row["memory_type"],
            "content": row["content"],
            "importance": row["importance"],
            "visibility": row["visibility"]
        }
        for row in rows
    ]


# ============================================================
# GEMINI RESPONSE
# ============================================================

async def generate_ai_response(
    user,
    user_message,
    channel_id,
    conversation_type,
    context=""
):
    """
    Generate an AI response using Gemini.
    """

    # These DB lookups are wrapped individually so that a
    # Postgres problem (bad DATABASE_URL, connection drop,
    # etc.) degrades to "no memory / no history" instead of
    # blowing up the whole response with a generic error.
    try:
        memories = await get_user_memories(
            user.id,
            limit=10
        )
    except Exception as error:
        print("=" * 60, flush=True)
        print("ERROR in get_user_memories:", flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        memories = []

    try:
        recent_messages = await get_recent_messages(
            user.id,
            channel_id=channel_id,
            conversation_type=conversation_type,
            limit=MAX_HISTORY_MESSAGES
        )
    except Exception as error:
        print("=" * 60, flush=True)
        print("ERROR in get_recent_messages:", flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        recent_messages = []

    memory_text = ""

    if memories:
        memory_text = "\n".join(
            f"- ({memory['visibility']}) {memory['content']}"
            for memory in memories
        )

    history_text = ""

    if recent_messages:
        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in recent_messages
        )

    prompt = f"""
USER ID:
{user.id}

USER NAME:
{user.display_name}

RELEVANT USER MEMORY:
{memory_text or "No stored memory."}

RECENT CONVERSATION:
{history_text or "No previous conversation available."}

CURRENT CONTEXT:
{context or "No additional context."}

CURRENT USER MESSAGE:
{user_message}
"""

    try:

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=AI_SYSTEM_INSTRUCTION
            )
        )

        if not response or not response.text:
            return (
                "I couldn't generate a response right now."
            )

        return response.text.strip()

    except Exception as error:

        logging.exception(
            "Gemini error: %s",
            error
        )

        return (
            "I ran into a problem while processing that. "
            "Try again in a moment."
        )


# ============================================================
# MEMORY DECISION
# ============================================================

async def classify_memory(
    information,
    user_id
):
    """
    Ask Gemini whether information is worth remembering.

    This is intentionally simple for the first version.
    """

    prompt = f"""
Decide whether the following information is worth keeping
as long-term memory for Discord user ID {user_id}.

Keep information permanently if it is likely to be useful
in future conversations.

Examples of useful long-term information:
- ongoing projects
- learning goals
- important preferences
- Python skill level
- important technical weaknesses
- important plans
- significant achievements

Do not permanently store:
- greetings
- random casual statements
- temporary questions
- ordinary conversation
- information useful only for the current message

Return ONLY JSON:

{{
    "remember": true,
    "importance": 1,
    "memory_type": "general",
    "content": "short memory"
}}

Importance must be between 1 and 5.

Information:
{information}
"""

    try:

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt
        )

        raw = response.text.strip()

        # Remove markdown fences if Gemini adds them.
        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        result = json.loads(raw)

        remember = bool(
            result.get("remember", False)
        )

        importance = int(
            result.get("importance", 1)
        )

        importance = max(
            1,
            min(5, importance)
        )

        memory_type = result.get(
            "memory_type",
            "general"
        )

        content = result.get(
            "content",
            information
        )

        return {
            "remember": remember,
            "importance": importance,
            "memory_type": memory_type,
            "content": content
        }

    except Exception as error:

        logging.exception(
            "Memory classification error: %s",
            error
        )

        return {
            "remember": False,
            "importance": 1,
            "memory_type": "general",
            "content": information
        }


# ============================================================
# AUTOMATIC MEMORY PROCESSING
# ============================================================

async def process_possible_memory(
    user,
    message_content
):
    """
    Decide whether a message contains information worth
    remembering permanently.
    """

    result = await classify_memory(
        message_content,
        user.id
    )

    if not result["remember"]:
        return

    await save_memory(
        user_id=user.id,
        memory_type=result["memory_type"],
        content=result["content"],
        importance=result["importance"],
        visibility="private"
    )

    logging.info(
        "Saved memory for user %s: %s",
        user.id,
        result["content"]
    )


# ============================================================
# CHALLENGE SYSTEM
# ============================================================

CHALLENGE_DIFFICULTIES = [
    "beginner",
    "easy",
    "intermediate",
    "hard",
    "advanced"
]


CHALLENGE_TOPICS = [
    "variables",
    "strings",
    "lists",
    "dictionaries",
    "loops",
    "functions",
    "exceptions",
    "classes",
    "object-oriented programming",
    "file handling",
    "modules",
    "decorators",
    "generators",
    "asyncio",
    "APIs",
    "algorithms",
    "data structures"
]


async def generate_daily_challenge():
    """
    Ask Gemini to generate one Python challenge.
    """

    difficulty = random.choice(
        CHALLENGE_DIFFICULTIES
    )

    topic = random.choice(
        CHALLENGE_TOPICS
    )

    prompt = f"""
Create one Python programming challenge.

Difficulty:
{difficulty}

Topic:
{topic}

The challenge should be practical and interesting.

Return ONLY valid JSON:

{{
    "title": "...",
    "description": "...",
    "difficulty": "{difficulty}",
    "topic": "{topic}",
    "solution": "..."
}}

Do not make the challenge trivial.
Do not make it impossible.
The solution should be a valid Python solution.
"""

    try:

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt
        )

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        challenge = json.loads(raw)

        return challenge

    except Exception as error:

        logging.exception(
            "Challenge generation error: %s",
            error
        )

        return {
            "title": "Python Logic Challenge",
            "description": (
                "Write a Python function that returns the "
                "second-largest unique number in a list."
            ),
            "difficulty": "intermediate",
            "topic": "lists",
            "solution": (
                "def second_largest(numbers):\n"
                "    unique = sorted(set(numbers))\n"
                "    return unique[-2]"
            )
        }


async def get_next_challenge_number():
    """
    Determine the next challenge number.
    """

    if postgres_pool is None:
        return 1

    async with postgres_pool.acquire() as connection:

        row = await connection.fetchrow(
            """
            SELECT COALESCE(
                MAX(challenge_number),
                0
            ) + 1 AS next_number
            FROM challenges
            """
        )

        return row["next_number"]


async def save_challenge(challenge):
    """
    Save a generated challenge to PostgreSQL.
    """

    if postgres_pool is None:
        return None

    challenge_number = (
        await get_next_challenge_number()
    )

    async with postgres_pool.acquire() as connection:

        row = await connection.fetchrow(
            """
            INSERT INTO challenges
            (
                challenge_number,
                title,
                description,
                difficulty,
                topic,
                solution
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, challenge_number
            """,
            challenge_number,
            challenge["title"],
            challenge["description"],
            challenge["difficulty"],
            challenge["topic"],
            challenge["solution"]
        )

        return row


async def get_latest_challenge():
    """
    Get the latest challenge from PostgreSQL.
    """

    if postgres_pool is None:
        return None

    async with postgres_pool.acquire() as connection:

        row = await connection.fetchrow(
            """
            SELECT *
            FROM challenges
            ORDER BY challenge_number DESC
            LIMIT 1
            """
        )

        return row


# ============================================================
# DAILY CHALLENGE POSTING
# ============================================================

# Set the channel IDs where daily challenges should be posted.
#
# Example:
#
# CHALLENGE_CHANNEL_IDS = {
#     123456789012345678,
#     987654321098765432
# }
#
# Leave empty until you add your actual channel IDs.

CHALLENGE_CHANNEL_IDS = set()


async def post_daily_challenge():
    """
    Generate, save, and post one challenge.
    """

    if not CHALLENGE_CHANNEL_IDS:
        logging.info(
            "No challenge channels configured."
        )
        return

    challenge = await generate_daily_challenge()

    saved = await save_challenge(
        challenge
    )

    if not saved:
        return

    challenge_number = saved["challenge_number"]

    embed = discord.Embed(
        title=(
            f"🐍 Python Challenge #{challenge_number}"
        ),
        description=challenge["description"],
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(
        name="Difficulty",
        value=challenge["difficulty"].title(),
        inline=True
    )

    embed.add_field(
        name="Topic",
        value=challenge["topic"].title(),
        inline=True
    )

    embed.set_footer(
        text="Come back tomorrow for the next challenge."
    )

    for channel_id in CHALLENGE_CHANNEL_IDS:

        channel = bot.get_channel(
            channel_id
        )

        if channel is None:
            continue

        try:
            await channel.send(
                embed=embed
            )

        except Exception as error:

            logging.exception(
                "Could not post challenge: %s",
                error
            )


# Fires once every day at a FIXED UTC clock time, instead of
# drifting based on when the bot process happened to start
# (the old @tasks.loop(hours=24) counted 24h from bot-ready,
# which shifts every restart).
@tasks.loop(
    time=dt_time(
        hour=DAILY_CHALLENGE_HOUR,
        minute=DAILY_CHALLENGE_MINUTE,
        tzinfo=timezone.utc
    )
)
async def daily_challenge():
    await post_daily_challenge()


@daily_challenge.before_loop
async def before_daily_challenge():
    """
    Wait until Discord is ready before starting
    the daily challenge task.
    """

    await bot.wait_until_ready()


# ============================================================
# ACTIVE CALL SYSTEM (scoped per user + channel)
# ============================================================

@bot.command(name="call")
async def call_command(ctx):
    """
    Activate continuous conversation for the user, scoped to
    this specific channel/DM only.
    """

    key = (ctx.author.id, ctx.channel.id)

    active_calls.add(key)

    await ctx.send(
        f"{ctx.author.mention} I'm listening in this channel. "
        f"Use `{BOT_PREFIX}stop` when you're done."
    )


@bot.command(name="stop")
async def stop_command(ctx):
    """
    Stop continuous conversation for the user in this
    channel/DM.
    """

    key = (ctx.author.id, ctx.channel.id)

    if key not in active_calls:

        await ctx.send(
            "You're not currently in an active AI conversation "
            "in this channel."
        )

        return

    active_calls.discard(key)

    await ctx.send(
        f"{ctx.author.mention} Alright, I'll stop listening here."
    )


# ============================================================
# CHALLENGE COMMAND
# ============================================================

@bot.command(name="challenge")
async def challenge_command(ctx):
    """
    Show the latest daily challenge.
    """

    challenge = await get_latest_challenge()

    if not challenge:

        await ctx.send(
            "There isn't a challenge yet."
        )

        return

    embed = discord.Embed(
        title=(
            f"🐍 Python Challenge "
            f"#{challenge['challenge_number']}"
        ),
        description=challenge["description"]
    )

    embed.add_field(
        name="Difficulty",
        value=challenge["difficulty"].title(),
        inline=True
    )

    embed.add_field(
        name="Topic",
        value=challenge["topic"].title(),
        inline=True
    )

    await ctx.send(
        embed=embed
    )


# ============================================================
# PING
# ============================================================

@bot.command(name="ping")
async def ping_command(ctx):
    """
    Basic bot test.
    """

    latency = round(
        bot.latency * 1000
    )

    await ctx.send(
        f"Pong! `{latency}ms`"
    )


# ============================================================
# STATUS
# ============================================================

@bot.command(name="status")
async def status_command(ctx):
    """
    Display basic bot status.
    """

    postgres_status = (
        "connected"
        if postgres_pool
        else "not connected"
    )

    await ctx.send(
        "Bot status:\n"
        f"Discord: connected\n"
        f"Gemini: connected\n"
        f"PostgreSQL: {postgres_status}\n"
        f"SQLite: connected"
    )


# ============================================================
# HELP
# ============================================================

@bot.command(name="help_ai")
async def help_command(ctx):
    """
    Show available basic commands.
    """

    await ctx.send(
        f"""
**Python AI Bot**

`{BOT_PREFIX}call`
Start a continuous conversation in this channel.

`{BOT_PREFIX}stop`
Stop the continuous conversation in this channel.

`{BOT_PREFIX}challenge`
Show the latest Python challenge.

`{BOT_PREFIX}ping`
Check bot latency.

`{BOT_PREFIX}status`
Check bot status.

You can also mention me or reply directly to one of my messages.
"""
    )


# ============================================================
# MESSAGE HANDLER
# ============================================================

@bot.event
async def on_message(message):
    """
    Main message processor.
    """

    # Ignore the bot's own messages.
    if message.author.bot:
        return

    # Process commands first.
    await bot.process_commands(message)

    # Do not process command messages as AI messages.
    if message.content.startswith(
        BOT_PREFIX
    ):
        return

    # --------------------------------------------------------
    # USER ID
    # --------------------------------------------------------

    user = message.author

    user_id = user.id

    # --------------------------------------------------------
    # USER DATABASE RECORD
    # --------------------------------------------------------

    try:
        await get_or_create_user(user)

    except Exception as error:

        logging.exception(
            "Could not create/get user: %s",
            error
        )

    # --------------------------------------------------------
    # DETERMINE WHETHER BOT SHOULD RESPOND
    # --------------------------------------------------------

    mentioned = bot.user in message.mentions

    replied_to_bot = False

    if message.reference:

        try:

            referenced_message = (
                message.reference.resolved
            )

            if referenced_message is None:
                # Not cached -- fetch it explicitly instead of
                # silently giving up.
                try:
                    referenced_message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                except Exception:
                    referenced_message = None

            if (
                referenced_message
                and referenced_message.author.id
                == bot.user.id
            ):
                replied_to_bot = True

        except Exception:
            replied_to_bot = False

    call_key = (user_id, message.channel.id)

    in_active_call = call_key in active_calls

    if not (
        mentioned
        or replied_to_bot
        or in_active_call
    ):
        return

    # --------------------------------------------------------
    # CLEAN MESSAGE
    # --------------------------------------------------------

    content = message.content

    if bot.user:

        content = content.replace(
            f"<@{bot.user.id}>",
            ""
        )

        content = content.replace(
            f"<@!{bot.user.id}>",
            ""
        )

    content = content.strip()

    if not content:
        await message.channel.send(
            "Yeah?"
        )
        return

    # --------------------------------------------------------
    # CONVERSATION TYPE
    # --------------------------------------------------------

    if isinstance(
        message.channel,
        discord.DMChannel
    ):
        conversation_type = "private"

    else:
        conversation_type = "public"

    # --------------------------------------------------------
    # GET OR CREATE CONVERSATION
    #
    # Reuses the existing open conversation for this
    # user+channel instead of inserting a new row every
    # message.
    # --------------------------------------------------------

    conversation_id = None

    try:

        conversation_id = await get_or_create_conversation(
            user_id=user_id,
            guild_id=(
                message.guild.id
                if message.guild
                else None
            ),
            channel_id=message.channel.id,
            conversation_type=conversation_type
        )

    except Exception as error:

        logging.exception(
            "Conversation creation failed: %s",
            error
        )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    if conversation_id:

        await save_message(
            conversation_id=conversation_id,
            discord_message_id=message.id,
            role="user",
            content=content
        )

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    try:

        async with message.channel.typing():

            response = await generate_ai_response(
                user=user,
                user_message=content,
                channel_id=message.channel.id,
                conversation_type=conversation_type,
                context=(
                    f"Discord server: "
                    f"{message.guild.name if message.guild else 'DM'}\n"
                    f"Channel: {message.channel.name if hasattr(message.channel, 'name') else 'DM'}\n"
                    f"Conversation type: {conversation_type}"
                )
            )

    except Exception as error:

        # Make sure a failure here is impossible to miss:
        # print a raw traceback to stdout (bypasses any
        # logging handler/level misconfiguration) AND log it
        # normally.
        print("=" * 60, flush=True)
        print("UNEXPECTED ERROR generating AI response:", flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        sys.stdout.flush()

        logging.exception(
            "Unexpected error generating AI response: %s",
            error
        )

        await message.channel.send(
            "Something went wrong while I was thinking about "
            "that. Try again in a moment."
        )

        return

    # --------------------------------------------------------
    # SEND RESPONSE
    # --------------------------------------------------------

    sent_message = None

    try:

        # Discord messages have a character limit.
        if len(response) <= 2000:

            sent_message = await message.channel.send(
                response,
                reference=message
            )

        else:

            # Split long AI responses.
            chunks = [
                response[i:i + 1900]
                for i in range(
                    0,
                    len(response),
                    1900
                )
            ]

            for chunk in chunks:

                sent_message = await message.channel.send(
                    chunk
                )

    except Exception as error:

        logging.exception(
            "Failed to send AI response: %s",
            error
        )

        return

    # --------------------------------------------------------
    # SAVE AI RESPONSE
    # --------------------------------------------------------

    if conversation_id and sent_message:

        await save_message(
            conversation_id=conversation_id,
            discord_message_id=sent_message.id,
            role="model",
            content=response
        )

    # --------------------------------------------------------
    # TEMPORARY CONTEXT (now non-blocking)
    # --------------------------------------------------------

    await save_temporary_data(
        user_id=user_id,
        data_type="last_response",
        data={
            "message": response,
            "channel_id": message.channel.id
        }
    )

    # --------------------------------------------------------
    # MEMORY CHECK
    # --------------------------------------------------------

    try:

        await process_possible_memory(
            user=user,
            message_content=content
        )

    except Exception as error:

        logging.exception(
            "Memory processing failed: %s",
            error
        )


# ============================================================
# READY EVENT
# ============================================================

@bot.event
async def on_ready():
    """
    Runs when the bot successfully connects to Discord.
    """

    logging.info(
        "Logged in as %s (%s)",
        bot.user,
        bot.user.id
    )

    logging.info(
        "Connected to %s guild(s).",
        len(bot.guilds)
    )

    # Start daily challenge task once.
    if not daily_challenge.is_running():

        daily_challenge.start()

    logging.info(
        "AI Python Discord Bot is ready."
    )


# ============================================================
# STARTUP
# ============================================================

async def startup():
    """
    Initialize databases and start the bot.
    """

    # SQLite
    await initialize_sqlite()

    # PostgreSQL
    await initialize_postgres()

    # Start Discord bot
    await bot.start(
        DISCORD_TOKEN
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            startup()
        )

    except KeyboardInterrupt:

        logging.info(
            "Bot stopped manually."
        )
