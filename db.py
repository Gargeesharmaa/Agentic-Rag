# db.py
import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")


def init_db():
    """Create the chat_logs table if it does not exist."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(255),
                    user_message TEXT,
                    bot_response TEXT
                );
            """
            )
            conn.commit()


def save_chat_to_db(session_id, user_message, bot_response):
    """Saves a conversation turn to PostgreSQL."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs (session_id, user_message, bot_response)
                VALUES (%s, %s, %s);
            """,
                (session_id, user_message, bot_response),
            )
            conn.commit()


def load_chat_from_db(session_id="gradio_default_session"):
    """Loads previous messages from PostgreSQL for the Gradio UI."""
    formatted_history = []
    try:
        with psycopg.connect(DB_URI, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_message, bot_response 
                    FROM chat_logs 
                    WHERE session_id = %s 
                    ORDER BY id ASC;
                """,
                    (session_id,),
                )
                rows = cur.fetchall()
                for row in rows:
                    formatted_history.append(
                        {"role": "user", "content": row["user_message"]}
                    )
                    formatted_history.append(
                        {"role": "assistant", "content": row["bot_response"]}
                    )
    except Exception as e:
        print(f"Error loading chat from DB: {e}")

    return formatted_history