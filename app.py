from datetime import datetime
import json
import gradio as gr
from agent_engine import master_agent
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = "chat_log.json"


def save_to_file(user_message, bot_response):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


def load_saved_history():
    formatted_history = []
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
            for entry in logs:
                formatted_history.append(
                    {"role": "user", "content": entry["user_message"]}
                )
                formatted_history.append(
                    {"role": "assistant", "content": entry["bot_response"]}
                )
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return formatted_history


def chat_response(message, history):
    try:
        config = {"configurable": {"thread_id": "gradio_session"}}

        response = master_agent.invoke(
            {"messages": [("human", message)]}, config=config
        )

        final_answer = response["messages"][-1].content

        save_to_file(message, final_answer)
        return final_answer
    except Exception as e:
        return f"An error occurred while processing your request:\n`{str(e)}`"

demo = gr.ChatInterface(
    fn=chat_response,
    chatbot=gr.Chatbot(
        value=load_saved_history(),
        height=550,
    ),
    title="Agentic RAG Assistant",
    description="Ask general corporate questions or request information from internal knowledge base documents.",
    examples=[
        "Why do we need ensemble models?",
        "What are our key company policies?",
        "Can you explain how random forests work?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)