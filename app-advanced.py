from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter


data = SimpleDirectoryReader(input_dir="./data", filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(
    data, 
    show_progress=True,
    transformations=[text_splitter]
    )
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

chat_history = []

while True:
    message = input("Enter your message: ")

    if message == "exit":
        break

    # Create user message
    user_message = ChatMessage(role=MessageRole.USER, content=message)
    chat_history.append(user_message)

    response = chat_engine.stream_chat(message, chat_history=chat_history)
    
    # Create assistant message from response
    assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content="")
    full_response = ""
    for token in response.response_gen:
        print(token, end="", flush=True)
        full_response += token

    # Add assistant message to chat history
    assistant_message.content = full_response
    chat_history.append(assistant_message)
    print("\n\n")

