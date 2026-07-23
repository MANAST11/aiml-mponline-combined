import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from local .env
load_dotenv()

# Fallback: Load environment variables from user's home directory if they ran the setup there
home_env = Path.home() / ".env"
if home_env.exists():
    load_dotenv(dotenv_path=home_env)

# Resolve API Key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if api_key:
    # Set both environment variables to ensure compatibility with all underlying libraries
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("WARNING: GEMINI_API_KEY is not set in environment, local .env, or home .env file.")

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure LlamaIndex to use Gemini LLM and local HuggingFace Embeddings
# Local embeddings avoid all API rate limit/quota issues (free and fast)
if api_key:
    Settings.llm = Gemini(model="models/gemini-3.5-flash", api_key=api_key)
else:
    Settings.llm = Gemini(model="models/gemini-3.5-flash")

# Load lightweight, high-performance local embedding model
print("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Local embedding model loaded successfully.")

app = FastAPI(title="PDF RAG Chatbot Backend")

# Setup directories
DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
STATIC_DIR = Path("./static")

DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Global index placeholder
index = None

def get_or_create_index():
    global index
    if index is not None:
        return index

    # Try to load existing index from storage
    if os.path.exists(STORAGE_DIR / "docstore.json"):
        try:
            print("Loading index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Error loading index (likely embedding dimension mismatch): {e}. Clearing storage and recreating empty index.")
            # Delete old index files to heal the storage directory
            for file_path in STORAGE_DIR.glob("*"):
                try:
                    file_path.unlink()
                except Exception as unlink_err:
                    print(f"Could not delete {file_path.name}: {unlink_err}")

    # Create a new empty index if no index exists
    print("Initializing empty index...")
    index = VectorStoreIndex([])
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index

# Initialize index on startup
@app.on_event("startup")
def startup_event():
    get_or_create_index()

class QueryRequest(BaseModel):
    message: str

@app.post("/api/query")
async def query_index(request: QueryRequest):
    print(f"\n--- [QUERY REQUEST] Message: '{request.message}' ---")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY is not set. Please configure it in your .env file."
        )
    
    try:
        current_index = get_or_create_index()
        # If there are no documents, guide the user to upload one
        if not list(DATA_DIR.glob("*.pdf")):
            return {
                "response": "Hello! I am ready to answer your questions, but there are no documents in my knowledge base yet. Please upload a PDF using the sidebar to get started.",
                "sources": []
            }

        # similarity_top_k=10 retrieves the top 10 most relevant document chunks
        query_engine = current_index.as_query_engine(similarity_top_k=10)
        response = query_engine.query(request.message)
        
        # Extract response text and source references
        sources = []
        print("Retrieved Chunks from Vector Store:")
        if hasattr(response, "source_nodes"):
            for idx, node_with_score in enumerate(response.source_nodes):
                node = node_with_score.node
                score = round(float(node_with_score.score or 0.0), 3)
                file_name = node.metadata.get("file_name", "Unknown File")
                page = node.metadata.get("page_label", "N/A")
                snippet = node.get_content().replace('\n', ' ')[:150]
                
                print(f"  [{idx+1}] File: {file_name} | Page: {page} | Score: {score}")
                print(f"      Snippet: {snippet}...")
                
                sources.append({
                    "file_name": file_name,
                    "page": page,
                    "snippet": node.get_content()[:200] + "...",
                    "score": score
                })

        print(f"LLM Response: '{str(response)[:200]}...'")
        print("--- [QUERY COMPLETE] ---\n")

        return {
            "response": str(response),
            "sources": sources
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"\n--- [UPLOAD REQUEST] Received: {file.filename} ---")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is missing!")
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY is not set. Please configure it in your .env file."
        )
        
    if not file.filename.endswith(".pdf"):
        print("ERROR: Rejected non-PDF file upload.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = DATA_DIR / file.filename
    
    # Save the file locally
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"File successfully saved to local directory: {file_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
    # Ingest file into LlamaIndex
    try:
        current_index = get_or_create_index()
        print("Reading and parsing PDF data via SimpleDirectoryReader...")
        
        # Load the uploaded file using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        documents = reader.load_data()
        print(f"PDF parsed. Extracted {len(documents)} document pages.")
        
        if not documents:
            raise ValueError("No text content could be extracted from this PDF. It might be scanned/image-only, or encrypted.")
        
        # Split document pages into smaller nodes (chunks)
        from llama_index.core.node_parser import SentenceSplitter
        print("Splitting document pages into smaller semantic chunks...")
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Generated {len(nodes)} semantic chunks for indexing.")
        
        # Insert nodes one-by-one with clear terminal logging
        print("Indexing document chunks into vector database (running locally on CPU)...")
        for idx, node in enumerate(nodes):
            print(f"-> Indexing chunk {idx+1}/{len(nodes)} ({(idx+1)/len(nodes)*100:.1f}%) ...")
            current_index.insert_nodes([node])
            
        print("Persisting updated index to storage...")
        current_index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        print("--- [UPLOAD SUCCESS] Successfully updated and persisted index! ---\n")
        
        return {
            "message": f"Successfully uploaded and indexed {file.filename}",
            "filename": file.filename
        }
    except Exception as e:
        import traceback
        print("ERROR: Failed during indexing process!")
        traceback.print_exc()
        
        # Clean up file if indexing fails
        if file_path.exists():
            print(f"Cleaning up/removing failed file: {file_path}")
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    try:
        files = []
        for file_path in DATA_DIR.glob("*.pdf"):
            files.append({
                "name": file_path.name,
                "size_kb": round(file_path.stat().st_size / 1024, 1)
            })
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list documents: {str(e)}")

@app.post("/api/clear")
async def clear_database():
    global index
    print("\n--- [RESET DATABASE REQUEST] Clearing all files and vectors ---")
    try:
        # Delete files in DATA_DIR
        for file_path in DATA_DIR.glob("*"):
            if file_path.is_file():
                print(f"Deleting raw file: {file_path.name}")
                file_path.unlink()
        
        # Delete files in STORAGE_DIR
        for file_path in STORAGE_DIR.glob("*"):
            if file_path.is_file():
                print(f"Deleting index file: {file_path.name}")
                file_path.unlink()
        
        # Re-initialize empty index
        print("Re-initializing empty index...")
        index = VectorStoreIndex([])
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        print("--- [RESET SUCCESS] Database cleared and re-initialized! ---\n")
        return {"message": "All documents and vectors cleared successfully."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

# Mount static files for the frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
