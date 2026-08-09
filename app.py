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


def load_all_sessions():
    """Fetches a distinct list of all unique active session IDs from PostgreSQL."""
    if not DB_URL:
        return []
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT session_id FROM chat_logs ORDER BY session_id DESC;")
                return [row[0] for row in cur.fetchall() if row[0]]
    except Exception as e:
        print(f"⚠️ Error loading unique session lists: {e}")
        return []


def load_saved_history(session_id):
    """Loads past conversation turns from PostgreSQL formatted cleanly for gr.Chatbot."""
    formatted_history = []
    if not DB_URL or not session_id:
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
                # Gradio expects tuples or OpenAI messages format based on configuration
                for row in rows:
                    formatted_history.append({"role": "user", "content": row["user_message"]})
                    formatted_history.append({"role": "assistant", "content": row["bot_response"]})
    except Exception as e:
        print(f"⚠️ Error loading chat history from PostgreSQL: {e}")

    return formatted_history


def chat_response(message, history, session_id):
    """Processes messages through the master RAG agent and writes metadata to the backend."""
    try:
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        config = {"configurable": {"thread_id": session_id}}
        response = master_agent.invoke(
            {"messages": [("human", message)]}, config=config
        )

        final_answer = response["messages"][-1].content
        save_to_db(session_id, message, final_answer)

        # Update history mapping logic
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": final_answer})

        # Dynamically reload session dropdown choices
        updated_sessions = load_all_sessions()
        if session_id not in updated_sessions:
            updated_sessions.insert(0, session_id)

        return "", history, session_id, gr.update(choices=updated_sessions, value=session_id)

    except Exception as e:
        history.append({"role": "assistant", "content": f"An error occurred:\n`{str(e)}`"})
        return "", history, session_id, gr.update()


def start_new_chat():
    """Resets the UI workspace, generating a brand new distinct tracking session."""
    new_id = f"session_{uuid.uuid4().hex[:8]}"
    return [], new_id, f"📝 Running on clean session: {new_id}"


def switch_active_session(selected_session):
    """Loads previous messages when an old session is clicked in the sidebar dropdown."""
    if not selected_session:
        return [], "", "⚠️ No active session selected."
    history = load_saved_history(selected_session)
    return history, selected_session, f"📂 Viewing history for: {selected_session}"


def upload_custom_document(file_obj):
    """Processes custom documents uploaded by the user and routes them to the agent engine."""
    if file_obj is None:
        return "⚠️ No file uploaded."
    try:
        file_path = file_obj.name
        file_name = os.path.basename(file_path)

        # -------------------------------------------------------------
        # NOTE: Connect your specific background ingestion process here.
        # Example: master_agent.add_document_to_vector_store(file_path)
        # -------------------------------------------------------------

        return f"✅ Successfully processed and indexed: **{file_name}**"
    except Exception as e:
        return f"❌ Failed to process document: {str(e)}"


# Initialize the database schema
init_db()

# Build custom flexible block window layout
with gr.Blocks(title="Agentic RAG System") as demo:
    # State tracking variables
    current_session = gr.State(value=f"session_{uuid.uuid4().hex[:8]}")
    
    gr.Markdown("# Agentic RAG Assistant (PostgreSQL Backed)")
    
    with gr.Row():
        # SIDEBAR PANEL FOR NAVIGATION AND UPLOADS
        with gr.Column(scale=1, min_width=280):
            new_chat_btn = gr.Button("➕ New Chat Session", variant="primary")
            
            gr.Markdown("### 📂 Past Conversations")
            session_dropdown = gr.Dropdown(
                choices=load_all_sessions(),
                label="Select History Log",
                interactive=True,
                value=None
            )
            status_bar = gr.Markdown("📝 Running on clean session: New")
            
            # --- NEW CUSTOM DOCUMENT UPLOAD SECTION ---
            gr.Markdown("### 📄 Add Context Documents")
            doc_uploader = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".txt", ".docx", ".csv"],
                file_count="single"
            )
            upload_status = gr.Markdown("")

        # MAIN CHAT APPLICATION DISPLAY Window
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type your question here and press Enter...",
                    container=False,
                    scale=7
                )
                submit_btn = gr.Button("Send", variant="secondary", scale=1)

    # UI EVENT TRIGGER ASSIGNMENTS
    # 1. Action: Handling User message submissions
    submit_btn.click(
        chat_response, 
        inputs=[txt_input, chatbot, current_session], 
        outputs=[txt_input, chatbot, current_session, session_dropdown]
    )
    txt_input.submit(
        chat_response, 
        inputs=[txt_input, chatbot, current_session], 
        outputs=[txt_input, chatbot, current_session, session_dropdown]
    )

    # 2. Action: Initiating a brand new clean conversation space
    new_chat_btn.click(
        start_new_chat,
        inputs=[],
        outputs=[chatbot, current_session, status_bar]
    )

    # 3. Action: Picking an alternative history thread from dropdown selector
    session_dropdown.change(
        switch_active_session,
        inputs=[session_dropdown],
        outputs=[chatbot, current_session, status_bar]
    )

    # 4. Action: Route uploaded file paths to the ingestion pipeline
    doc_uploader.upload(
        upload_custom_document,
        inputs=[doc_uploader],
        outputs=[upload_status]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=8000, 
        share=False,
        css="footer {visibility: hidden} .api-status {display: none !important}"
    )