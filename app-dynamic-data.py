from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
import os.path
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
Settings.text_splitter = text_splitter

# Define storage directory and state file
STORAGE_DIR = "./storage"
STATE_FILE = os.path.join(STORAGE_DIR, "data_state.json")
DATA_DIR = "./data"

def get_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_current_state() -> Dict[str, Any]:
    """Get current state of all files in the data directory."""
    state = {}
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, DATA_DIR)
            state[rel_path] = {
                "hash": get_file_hash(filepath),
                "modified_time": os.path.getmtime(filepath)
            }
    return state

def load_previous_state() -> Dict[str, Any]:
    """Load previous state from state file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state: Dict[str, Any]) -> None:
    """Save current state to state file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def has_data_changed() -> bool:
    """Check if data directory has changed since last indexing."""
    current_state = get_current_state()
    previous_state = load_previous_state()
    
    # Check if any files were added, removed, or modified
    if set(current_state.keys()) != set(previous_state.keys()):
        return True
    
    # Check if any existing files were modified
    for file_path, current_info in current_state.items():
        if file_path not in previous_state:
            return True
        if current_info["hash"] != previous_state[file_path]["hash"]:
            return True
    
    return False

# Check if storage directory exists, if not create it
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Check if data has changed or if index doesn't exist
needs_reindex = has_data_changed() or not os.path.exists(os.path.join(STORAGE_DIR, "docstore.json"))

if needs_reindex:
    print("Changes detected in data directory or no existing index. Creating new index...")
    # Load documents and create index
    data = SimpleDirectoryReader(input_dir=DATA_DIR, filename_as_id=True).load_data()
    index = VectorStoreIndex.from_documents(
        data, 
        show_progress=True,
        transformations=[text_splitter]
    )
    # Persist index to disk
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    # Save current state
    save_state(get_current_state())
    print("Index created and stored successfully")
else:
    print("Loading existing index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully")

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

