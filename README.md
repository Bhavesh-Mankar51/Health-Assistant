# CarePlus

A Retrieval-Augmented Generation (RAG) based health assistant that leverages advanced AI and vector databases to provide medical information and health guidance through a conversational interface.

## Overview

CarePlus is an intelligent medical question-answering system built with modern AI technologies. It combines:
- **LangChain** for orchestrating LLM workflows
- **OpenAI GPT-4** for intelligent medical response generation
- **Pinecone Vector Database** for semantic search and knowledge retrieval
- **Sentence Transformers** for text embedding
- **Flask** for web application framework

## Technical Architecture

### Core Components

1. **RAG Pipeline (Retrieval-Augmented Generation)**
   - Retrieves relevant medical documents using vector similarity search
   - Generates contextual responses based on retrieved information
   - Ensures answers are grounded in actual medical knowledge

2. **Vector Database (Pinecone)**
   - Stores embeddings of medical documents
   - Enables fast semantic search with cosine similarity metrics
   - Configured index: `health-assistant` (384-dimensional embeddings, serverless spec on AWS)

3. **Embedding Model**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for converting text to 384-dimensional vectors
   - Provides semantic understanding of medical content

4. **LLM (Large Language Model)**
   - OpenAI GPT-4o for generating natural language responses
   - Configured with system prompt for medical context
   - Limits responses to 3 sentences for concise information

5. **Web Application**
   - Flask-based backend serving REST API
   - HTML/CSS/JavaScript frontend for chat interface
   - Real-time communication between frontend and RAG chain

## Technology Stack

### Core Dependencies
- **langchain** (0.3.26) - LLM orchestration and RAG framework
- **langchain-openai** (0.3.24) - OpenAI integration
- **langchain-pinecone** (0.2.8) - Pinecone vector store integration
- **langchain-community** (0.3.26) - Community integrations
- **flask** (3.1.1) - Web framework
- **sentence-transformers** (4.1.0) - Text embeddings
- **pypdf** (5.6.1) - PDF document processing
- **python-dotenv** (1.1.0) - Environment variable management
- **pinecone-client** - Vector database client

## Project Structure

```
CarePlus/
├── app.py                    # Flask application and RAG chain setup
├── store_index.py            # Script to create and populate Pinecone index
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
├── .env                      # Environment variables (not in repo)
├── src/
│   ├── __init__.py
│   ├── helper.py             # PDF loading, text splitting, embedding functions
│   └── prompt.py             # System prompt for medical assistant
├── data/
│   └── *.pdf                 # Medical documents for indexing
├── templates/
│   └── chat.html             # Chat interface HTML
├── static/                   # CSS, JavaScript, images
└── research/                 # Research notes and documentation
```

## Key Modules

### app.py
- Initializes Flask web server on port 8080
- Sets up RAG chain with retriever (k=3 similar documents)
- Handles `/` route for serving chat interface
- Handles `/get` POST endpoint for chat messages
- Retrieves relevant medical documents and generates responses

### store_index.py
- Loads PDF files from `data/` directory
- Splits documents into 500-character chunks with 20-character overlap
- Generates embeddings using HuggingFace model
- Creates Pinecone serverless index if it doesn't exist
- Populates vector database with embedded documents

### src/helper.py
- `load_pdf_files()` - Loads all PDF files from directory
- `filter_to_minimal_docs()` - Preserves only essential metadata
- `text_split()` - Uses RecursiveCharacterTextSplitter for smart chunking
- `download_embeddings()` - Initializes HuggingFace embedding model

### src/prompt.py
- Defines system prompt for medical assistant
- Instructs model to be concise (max 3 sentences)
- Handles unknown answers appropriately

## Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhavesh-Mankar51/CarePlus.git
   cd CarePlus
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create `.env` file in project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

5. **Prepare medical documents**
   - Place PDF files in `data/` directory
   - These will be processed and indexed

6. **Create Pinecone index**
   ```bash
   python store_index.py
   ```
   - Loads PDFs, creates embeddings
   - Creates Pinecone index (if not exists)
   - Uploads vectors to database

7. **Run the application**
   ```bash
   python app.py
   ```
   - Server starts on http://localhost:8080
   - Open browser and access chat interface

## Usage

1. Access the web interface at `http://localhost:8080`
2. Enter your health-related question in the chat
3. The RAG system will:
   - Search Pinecone for relevant documents (top 3)
   - Pass retrieved context to GPT-4o
   - Generate a concise medical response
4. View the response in the chat interface

## How RAG Works

1. **Indexing Phase** (`store_index.py`)
   - Extract text from PDF documents
   - Split into semantic chunks
   - Generate embeddings for each chunk
   - Store in Pinecone with metadata

2. **Query Phase** (`app.py`)
   - User submits health question
   - Generate embedding for question
   - Search Pinecone for similar documents
   - Create prompt with retrieved context
   - Send to GPT-4o for response
   - Return answer to user

## Performance Considerations

- **Chunk Size**: 500 characters with 20-char overlap balances context and relevance
- **Embedding Model**: All-MiniLM-L6-v2 provides good accuracy with small footprint
- **Similarity Search**: Using cosine metric for semantic matching
- **Response Limit**: 3 sentences keeps answers focused and actionable

## Environment Variables

| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | Your OpenAI API key for GPT-4o access |
| PINECONE_API_KEY | Your Pinecone API key for vector database |

## API Endpoints

### GET / 
Returns the chat interface HTML page.

### POST /get
Processes user message and returns AI response.

**Request:**
```
POST /get
Content-Type: application/x-www-form-urlencoded

msg=What are symptoms of common cold?
```

**Response:**
```
Concise medical answer based on indexed documents...
```

## Author
Bhavesh Mankar (bhaveshmankar024@gmail.com)

## License
See LICENSE file for details