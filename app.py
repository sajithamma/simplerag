
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import datetime

# Configure your API Key and model securely

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

data = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(data)
#memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

print("Index loaded successfully")



llm = OpenAI(model="gpt-4o", api_key=api_key)
        
chat_engine = index.as_chat_engine(
    memory=None,
    llm=llm,
    context_prompt=(
        "You are a helpful assistant"
    ),
    verbose=False,
)

while True:
    message = input("Enter your message: ")

    if message == "exit":
        break

    response = chat_engine.stream_chat(message)
    for token in response.response_gen:  # Assuming response_gen is synchronous, adapt if needed
        print(token, end="", flush=True)

    print("\n\n")

