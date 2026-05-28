import os
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma-4-26b-a4b-it"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

system_instruction_preserve = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in a code-switching style mixing Indonesian, English, and Arabic naturally.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

Your answers must:
- Preserve and mirror the code-switching pattern of the user (mix Indonesian, English, and Arabic naturally).
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.

Example tone:
User: Cuaca hari ini gimana, bro?
Assistant: Hari ini cerah, bro. The temperature is around 30 degrees, jadi siapkan air minum ya.
"""

system_instruction_normalize = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

Your answers must:
- Be written in polite and easily understandable Indonesian only, regardless of the language the user uses.
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.
"""

client = genai.Client(api_key=GOOGLE_API_KEY)
history_adapter = TypeAdapter(list[types.Content])

def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")

def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)

def load_chat_history(config):
    if not os.path.exists(CHAT_HISTORY_FILE):
        return client.chats.create(model=MODEL, config=config)

    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model=MODEL, config=config)

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return client.chats.create(model=MODEL, config=config)

    try:
        history = history_adapter.validate_json(json_str)
        return client.chats.create(model=MODEL, config=config, history=history)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        return client.chats.create(model=MODEL, config=config)

def generate_response(prompt: str, mode: str = "normalize") -> str:
    if mode == "preserve":
        instruction = system_instruction_preserve
    else:
        instruction = system_instruction_normalize

    config = types.GenerateContentConfig(system_instruction=instruction)

    try:
        chat = load_chat_history(config)
        response = chat.send_message(prompt)
        save_chat_history(chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"