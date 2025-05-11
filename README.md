# SimpleRAG

A simple RAG (Retrieval-Augmented Generation) implementation using LlamaIndex and OpenAI. This project demonstrates different approaches to building a RAG system with increasing levels of sophistication.

## Features

- Basic RAG implementation with document indexing and chat interface
- Advanced version with text splitting and chat history
- Dynamic data version with automatic reindexing on data changes
- Persistent storage of vector indices
- Streaming responses for better user experience

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone git@github.com:sajithamma/simplerag.git
cd simplerag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

The project contains three main implementations:

1. `app.py` - Basic RAG implementation
2. `app-advanced.py` - Advanced version with text splitting and chat history
3. `app-dynamic-data.py` - Full version with automatic reindexing

### Directory Structure
```
simplerag/
├── data/               # Directory for your documents
├── storage/           # Directory for storing vector indices
├── app.py            # Basic implementation
├── app-advanced.py   # Advanced implementation
├── app-dynamic-data.py # Full implementation
└── requirements.txt   # Project dependencies
```

## Usage

### Basic Version (app.py)
Simple RAG implementation with basic document indexing:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Load and index documents
data = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(data)

# Create chat engine
chat_engine = index.as_chat_engine(
    llm=OpenAI(model="gpt-4"),
    verbose=False
)

# Chat interface
response = chat_engine.stream_chat("Your question here")
```

### Advanced Version (app-advanced.py)
Includes text splitting and chat history:
```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Configure text splitting
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

# Create index with text splitting
index = VectorStoreIndex.from_documents(
    data,
    transformations=[text_splitter]
)

# Chat with history
chat_history = []
user_message = ChatMessage(role=MessageRole.USER, content="Your question")
chat_history.append(user_message)
response = chat_engine.stream_chat(user_message.content, chat_history=chat_history)
```

### Dynamic Data Version (app-dynamic-data.py)
Includes automatic reindexing when data changes:
```python
# Check if data has changed
needs_reindex = has_data_changed() or not os.path.exists("storage/docstore.json")

if needs_reindex:
    # Create new index
    index = VectorStoreIndex.from_documents(data)
    index.storage_context.persist(persist_dir="storage")
else:
    # Load existing index
    index = load_index_from_storage(storage_context)
```

## Running the Application

1. Place your documents in the `data/` directory
2. Run any of the implementations:
```bash
# Basic version
python app.py

# Advanced version
python app-advanced.py

# Dynamic data version
python app-dynamic-data.py
```

3. Start chatting with your documents!

## Features by Version

### Basic Version (app.py)
- Simple document indexing
- Basic chat interface
- Streaming responses

### Advanced Version (app-advanced.py)
- Text splitting for better context
- Chat history support
- Persistent index storage
- Improved response quality

### Dynamic Data Version (app-dynamic-data.py)
- All advanced features
- Automatic reindexing on data changes
- File state tracking
- Efficient index management
