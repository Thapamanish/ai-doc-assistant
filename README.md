# AI Document Assistant

An intelligent document analysis tool that helps you find answers from your PDF documents using semantic search and AI-powered question answering.

## What It Does

Upload your PDFs, ask questions in plain English, and get accurate answers pulled directly from your documents. The system understands context and meaning, not just keywords.

## Key Features

- **PDF Support**: Upload and process multiple PDF documents
- **Smart Search**: Finds relevant information using semantic understanding
- **Contextual Answers**: Generates responses based on actual document content
- **Conversation Tracking**: Keeps a history of your questions and answers
- **Flexible Settings**: Adjust processing parameters for optimal results
- **Clean Interface**: Simple, intuitive design built with Streamlit

## Tech Stack

- Python 3.12
- Streamlit (UI framework)
- LangChain (document processing & RAG)
- FAISS (vector similarity search)
- Sentence Transformers (text embeddings)
- OpenAI GPT (answer generation)
- PyPDF (PDF processing)

## Requirements

- Python 3.12+
- OpenAI API key
- 2GB+ RAM (for model loading)

## Setup Instructions

**1. Clone this repository**
```bash
git clone <your-repo-url>
cd ai-document-assistant
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

**5. Launch the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload Documents**: Click the file uploader in the sidebar and select your PDF files
2. **Configure Settings** (optional): Adjust chunk size and overlap in the sidebar
3. **Process**: Click "Process Document" to prepare your documents for querying
4. **Ask Questions**: Type your question in the text box and press Enter
5. **Review Answers**: Read the AI-generated response along with source references

## How It Works

The system uses Retrieval Augmented Generation (RAG):

1. **Text Extraction**: Pulls text from your PDF documents
2. **Chunking**: Breaks text into overlapping segments for better context
3. **Embedding**: Converts text chunks into numerical vectors
4. **Indexing**: Stores vectors in FAISS for fast similarity search
5. **Retrieval**: Finds relevant chunks when you ask a question
6. **Generation**: Feeds context to GPT to generate accurate answers

## Configuration Options

- **Chunk Size**: Controls how much text goes into each segment (default: 1000 characters)
- **Chunk Overlap**: How much text overlaps between chunks (default: 200 characters)

Larger chunks = more context but slower processing  
Smaller chunks = faster but may miss connections

## Development

Install development tools:
```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
flake8 .
```

## Troubleshooting

**"Module not found" errors**: Make sure your virtual environment is activated and dependencies are installed

**Slow processing**: Reduce chunk size or process fewer documents at once

**API errors**: Check that your OpenAI API key is valid and has available credits

## License

MIT License - feel free to use this project however you'd like.

## Built With

- OpenAI API
- Streamlit framework
- FAISS vector database
- Sentence Transformers
- LangChain library