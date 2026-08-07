import os
from datetime import datetime
import gradio as gr
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
import uuid

from agent_engine import master_agent

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


def init_db():
    """Ensure the chat_logs table exists in PostgreSQL."""
    if not DB_URL:
        return
    try:
        with psycopg.connect(DB_URL) as conn:
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
    except Exception as e:
        print(f"⚠️ Failed to initialize Postgres database: {e}")


def save_to_db(session_id, user_message, bot_response):
    """Saves a conversation turn into PostgreSQL."""
    if not DB_URL:
        return
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_logs (session_id, user_message, bot_response)
                    VALUES (%s, %s, %s);
                """,
                    (session_id, user_message, bot_response),
                )
                conn.commit()
    except Exception as e:
        print(f"⚠️ Error saving log to PostgreSQL: {e}")


def load_saved_history(session_id="gradio_default_session"):
    """Loads past conversation turns from PostgreSQL for the UI."""
    formatted_history = []
    if not DB_URL:
        return formatted_history

    try:
        with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
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
        print(f"⚠️ Error loading chat history from PostgreSQL: {e}")

    return formatted_history

def get_or_create_session_id(session_id):
    """Generates a unique UUID session ID if one doesn't exist yet for the client."""
    if not session_id:
        session_id = f"session_{uuid.uuid4()}"
    return session_id


def chat_response(message, history, request: gr.Request):
    try:
        session_id = get_or_create_session_id(session_id)
        config = {"configurable": {"thread_id": session_id}}
        response = master_agent.invoke(
            {"messages": [("human", message)]}, config=config
        )

        final_answer = response["messages"][-1].content

        # Save turn to PostgreSQL
        save_to_db(session_id, message, final_answer)
        return final_answer

    except Exception as e:
        return f"An error occurred while processing your request:\n`{str(e)}`"

init_db()

demo = gr.ChatInterface(
    fn=chat_response,
    type="messages",
    chatbot=gr.Chatbot(
        value=load_saved_history(),
        height=550,
    ),
    title="Agentic RAG Assistant (PostgreSQL Backed)",
    description="Ask general corporate questions or request information from internal knowledge base documents.",
    examples=[
        "Why do we need ensemble models?",
        "Can you explain how random forests work?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=0, share=False)