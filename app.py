import flask
from flask import request, jsonify
from flask_cors import CORS # Import CORS
import json
import os
import re
import uuid
import logging
import tiktoken # Keep tiktoken for chunking if using token-based chunker

import chromadb
from chromadb.utils import embedding_functions # Use Chroma's helper

from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (especially OpenAI API key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Ensure this matches your .env file
if not openai_api_key:
    logging.error("OpenAI API key not found. Make sure it's set in the .env file as OPENAI_API_KEY.")
    exit(1)

# --- Constants ---
# *** MAKE SURE THIS PATH IS CORRECT FOR YOUR STRUCTURE ***
DATA_FILE = 'rag_backend/dustin_photography_reviews_cleaned.json'
# ChromaDB setup
CHROMA_PERSIST_DIR = "chroma_db" # Directory for OpenAI embeddings
COLLECTION_NAME = "photography_reviews" # Collection name for OpenAI embeddings
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Or "text-embedding-ada-002"
LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "gpt-4" or other compatible models

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    exit(1)

# --- Initialize ChromaDB ---
# Use OpenAI's embedding function directly with ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=EMBEDDING_MODEL_NAME
            )

# Initialize ChromaDB client with persistence
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
except Exception as e:
     logging.error(f"Failed to initialize ChromaDB client at '{CHROMA_PERSIST_DIR}': {e}")
     exit(1)


# --- Data Loading and Embedding (Run Once at Startup) ---
# Using the character-based chunker from Gemini example for simplicity,
# but you could adapt the token-based one if preferred.
def simple_chunker(text, max_chars=1800, overlap=200):
    """Basic chunker splitting by paragraph, then by sentence, then by word."""
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + max_chars, len(text))
        para_break = text.rfind('\n\n', current_pos, end_pos)
        if para_break > current_pos + overlap:
             end_pos = para_break + 2
        elif text.rfind('.', current_pos, end_pos) > current_pos + overlap:
             end_pos = text.rfind('.', current_pos, end_pos) + 1

        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end_pos - overlap
        current_pos = max(current_pos + 1, next_start)
        if next_start <= current_pos and end_pos == len(text):
             break
    return [c for c in chunks if c]

def setup_vector_db():
    """Loads data, chunks it, embeds it (via Chroma), and stores it."""
    logging.info("Setting up vector database for OpenAI...")
    try:
        # Get or create the collection, specifying the OpenAI embedding function
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef # Let Chroma handle OpenAI embedding
        )
        logging.info(f"Using ChromaDB collection: '{COLLECTION_NAME}'")

        if collection.count() > 0:
            logging.info(f"Collection '{COLLECTION_NAME}' already populated ({collection.count()} items). Skipping embedding.")
            return collection

        logging.info(f"Collection '{COLLECTION_NAME}' is empty. Loading and embedding data...")
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
        except FileNotFoundError:
            logging.error(f"Data file '{DATA_FILE}' not found. Check the path in app.py.")
            return None
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from '{DATA_FILE}'.")
            return None

        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for i, review in enumerate(reviews):
            content = review.get('content_text', '')
            url = review.get('url', '')
            title = review.get('title', 'Unknown Title')

            if not content:
                logging.warning(f"Skipping review {i+1} due to empty content (URL: {url})")
                continue

            chunks = simple_chunker(content) # Use the chunker
            logging.info(f"Review {i+1} ('{title}') chunked into {len(chunks)} parts.")

            for chunk_index, chunk in enumerate(chunks):
                doc_id = f"review_{i+1}_chunk_{chunk_index+1}"
                documents_to_add.append(chunk)
                metadatas_to_add.append({
                    "source_url": url,
                    "title": title,
                    "chunk_num": chunk_index + 1,
                     "original_doc_index": i+1
                })
                ids_to_add.append(doc_id)

        # Add documents to ChromaDB in batches. Chroma handles embedding.
        batch_size = 100 # OpenAI embedding endpoint often has batch limits
        for i in range(0, len(documents_to_add), batch_size):
            batch_docs = documents_to_add[i:i+batch_size]
            batch_metas = metadatas_to_add[i:i+batch_size]
            batch_ids = ids_to_add[i:i+batch_size]

            logging.info(f"Adding batch {i//batch_size + 1} ({len(batch_ids)} documents) to Chroma collection (will be embedded by Chroma)...")
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            # Optional delay if hitting rate limits
            # time.sleep(1)

        logging.info(f"Successfully added {len(documents_to_add)} chunks to the collection.")
        return collection

    except Exception as e:
        logging.exception(f"An error occurred during vector DB setup: {e}")
        return None

# --- Perform Setup at Application Start ---
collection = setup_vector_db()
if collection is None:
     logging.error("Failed to setup vector database. Exiting.")
     exit(1)

# --- Initialize Flask App ---
app = flask.Flask(__name__)
CORS(app) # Enable CORS

# --- RAG Function ---
def perform_rag(query, n_results=3):
    """Retrieves relevant context and generates an answer using OpenAI."""
    if not collection:
         return "Error: Vector database not initialized."

    # --- >>> START DEBUG PRINTING <<< ---
    print("\n--- RAG Process Started ---")
    print(f"[*] User Query Received: '{query}'")
    # --- >>> END DEBUG PRINTING <<< ---

    logging.info(f"Performing RAG for query: '{query}'")
    try:
        # 1. Retrieve relevant documents from ChromaDB
        # ChromaDB uses the collection's embedding function automatically for the query
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Include distances for debugging relevance
        )

        context_chunks = results['documents'][0] if results and results.get('documents') else []
        metadatas = results['metadatas'][0] if results and results.get('metadatas') else []
        distances = results['distances'][0] if results and results.get('distances') else []


        # --- >>> START DEBUG PRINTING <<< ---
        print("\n[*] Context Retrieval Results:")
        if not context_chunks:
            print("  - No relevant documents found.")
            context_str = "No relevant context found in the reviews." # Provide this to LLM
        else:
            print(f"  - Retrieved {len(context_chunks)} chunks:")
            context_lines = []
            for i, (doc, meta, dist) in enumerate(zip(context_chunks, metadatas, distances)):
                print(f"  --- Chunk {i+1} ---")
                print(f"    Source: {meta.get('title', 'N/A')}")
                print(f"    URL: {meta.get('source_url', 'N/A')}")
                print(f"    Distance: {dist:.4f}") # Lower distance = more similar
                print(f"    Content Snippet: {doc[:200]}...") # Print start of chunk
                context_lines.append(f"Source: {meta.get('title', 'N/A')}\nContent:\n{doc}")
            context_str = "\n\n---\n\n".join(context_lines)
        print("--- End Context ---")
        # --- >>> END DEBUG PRINTING <<< ---


        # 2. Construct the prompt for the LLM
        system_prompt = """You are a helpful assistant answering questions based ONLY on the provided context from Dustin Abbott's photography reviews.
If the answer is not found in the context, say 'I cannot answer this based on the provided reviews.' Do not make up information.
Be concise and directly answer the question using the information from the reviews."""

        user_prompt = f"""Based ONLY on the context below from Dustin Abbott's reviews, please answer the question.

Context:
--- START CONTEXT ---
{context_str}
--- END CONTEXT ---

Question: {query}

Answer:"""

        # --- >>> START DEBUG PRINTING <<< ---
        print("\n[*] Prompt being sent to OpenAI LLM:")
        print("--- System Prompt ---")
        print(system_prompt)
        print("--- User Prompt (including context) ---")
        print(user_prompt)
        print("--- End Prompt ---")
        # --- >>> END DEBUG PRINTING <<< ---

        # 3. Call OpenAI LLM
        logging.info("Sending prompt to OpenAI LLM...")
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        answer = response.choices[0].message.content
        logging.info("Received response from LLM.")

        # --- >>> START DEBUG PRINTING <<< ---
        print("\n[*] LLM Response Received:")
        print(answer)
        print("--- RAG Process Ended ---\n")
        # --- >>> END DEBUG PRINTING <<< ---

        return answer.strip()

    except Exception as e:
        logging.exception(f"An error occurred during RAG processing: {e}")
        print(f"\n[!!!] RAG Error: {e}\n") # Print error to terminal too
        return f"Sorry, an error occurred while processing your request."


# --- API Endpoint ---
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Receives user message, performs RAG, returns LLM response."""
    logging.info("Received request at /api/chat")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        logging.warning("Request received without 'message' field.")
        return jsonify({"error": "Missing 'message' field in request"}), 400

    # Perform RAG to get the answer
    llm_response = perform_rag(user_message)

    # Return the response
    return jsonify({"reply": llm_response})

# --- Run the Flask App ---
if __name__ == '__main__':
     # Check if the chroma directory exists, if not, embedding will run.
    if not os.path.exists(CHROMA_PERSIST_DIR):
        logging.warning(f"ChromaDB persistence directory '{CHROMA_PERSIST_DIR}' not found.")
        logging.warning("Database will be created and data will be embedded on first run.")
        logging.warning("This may take several minutes and incur API costs.")
    app.run(host='0.0.0.0', port=5000, debug=True) # Use debug=False for production