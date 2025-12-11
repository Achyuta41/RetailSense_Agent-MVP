import os
from google.genai import Client

# Load your Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set your GEMINI_API_KEY environment variable")

# Initialize client
client = Client(api_key=api_key)

# Create a chat session with a valid model
chat = client.chats.create(model="gemini-2.5-flash")

# Send a message
response = chat.send_message("Explain agentic AI in simple terms.")
print("Gemini says:\n", response.text)

# Optional: send a follow-up message
followup = chat.send_message("Summarize that in one sentence.")
print("Gemini follow-up:\n", followup.text)
