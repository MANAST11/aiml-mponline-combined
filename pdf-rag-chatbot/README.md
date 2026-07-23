# DocuMind AI: PDF RAG Chatbot

DocuMind AI is a lightweight, high-performance Retrieval-Augmented Generation (RAG) chatbot designed to run efficiently on standard laptops. By offloading heavy text generation to the cloud while computing text embeddings locally, it provides a fast, accurate, and completely free-tier friendly assistant for querying PDF documents.

---

## Key Features

*   **Hybrid Cloud-Local Architecture**: Uses cloud-based **Gemini 3.5 Flash** for rapid text generation and local **HuggingFace Embedding (BAAI/bge-small-en-v1.5)** to completely bypass API rate limits.
*   **Premium Web Interface**: Clean, modern dark-themed glassmorphism interface styled in pure vanilla CSS. Includes drag-and-drop file upload and typing indicators.
*   **Dynamic Citations**: Identifies and cites the exact PDF names and page numbers used to formulate answers. Users can click citation badges to read the raw source text snippets.
*   **Self-Healing Vector Index**: Automatically resets and updates storage schemas if mismatching embedding dimensions or database corruptions are detected.
*   **Real-time Process Logs**: Prints terminal progress bars during PDF splitting and local indexing, preventing frozen UI states.

---

## System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │                  User Input Query                      │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │ Local HuggingFace Embedding Encoder  │
                       │     (BAAI/bge-small-en-v1.5)         │
                       └──────────────────┬───────────────────┘
                                          │ (Creates Vector)
                                          ▼
                       ┌──────────────────────────────────────┐
                       │  Vector Index Similarity Lookup      │
                       │       (Local Vector Database)        │
                       └──────────────────┬───────────────────┘
                                          │ (Top 10 Context Chunks)
                                          ▼
                       ┌──────────────────────────────────────┐
                       │        Gemini 3.5 Flash API          │
                       │     (Cloud Text Generation)          │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │          Final Grounded Answer with Citations          │
               └────────────────────────────────────────────────────────┘
```

---

## Setup and Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Obtain a Free Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Create and copy your API Key.

### 3. Clone and Configure
1. Clone or download this repository.
2. In the root directory, create a file named `.env` and configure your API key as follows:
   ```env
   GEMINI_API_KEY=your_copied_api_key_here
   ```
   *(Note: You can copy and rename the `.env.example` file provided in the repo).*

### 4. Install Dependencies
Navigate to the project folder and install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## Running the Application

1. Start the local server:
   ```bash
   python app.py
   ```
2. Wait for the server console to display:
   ```text
   Local embedding model loaded successfully.
   Uvicorn running on http://127.0.0.1:8000
   ```
3. Open **`http://localhost:8000`** in your browser.
4. Drag and drop your PDF into the sidebar, watch the terminal for indexing progress, and start chatting!

---

## Technologies Used

*   **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Orchestration Framework**: [LlamaIndex](https://www.llamaindex.ai/)
*   **LLM Provider**: [Google Gemini API](https://ai.google.dev/) (Gemini 3.5 Flash)
*   **Embedding Model**: [HuggingFace Transformers](https://huggingface.co/) (`BAAI/bge-small-en-v1.5`)
*   **Frontend UI**: Vanilla HTML5, Vanilla CSS3 (Custom Glassmorphism), and Vanilla JavaScript
