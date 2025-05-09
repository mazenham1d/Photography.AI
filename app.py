# filename: app.py
import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import torch
from sentence_transformers import CrossEncoder
import numpy as np
import re

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize CrossEncoder for reranking (will be lazy-loaded when needed)
reranker = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A good general-purpose reranker model

def load_reranker():
    """Lazy-loads the reranker model when needed"""
    global reranker
    if reranker is None:
        logging.info(f"Loading reranker model: {RERANKER_MODEL}")
        try:
            reranker = CrossEncoder(RERANKER_MODEL)
            logging.info("Reranker model successfully loaded")
        except Exception as e:
            logging.error(f"Error loading reranker model: {e}")
            return None
    return reranker

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- ChromaDB Setup ---
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2" # Default embedding model (if needed)
COLLECTION_NAME = "photography_reviews"

# Initialize client and collection variables
client = None
collection = None
openai_ef = None

# Use OpenAI embeddings if API key is available
if os.getenv("OPENAI_API_KEY"):
    logging.info("Setting up vector database for OpenAI...")
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002" # Make sure this model name is correct
        )
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

        # Get or create the collection
        logging.info(f"Getting/Creating ChromaDB collection: '{COLLECTION_NAME}'")
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef # Use the specific embedding function instance
            # You might need metadata={"hnsw:space": "cosine"} depending on Chroma version/needs
        )
        logging.info(f"Successfully connected to collection '{COLLECTION_NAME}'.")

    except Exception as e:
        logging.error(f"Error setting up ChromaDB with OpenAI: {e}")
        # Decide how to handle this: raise error, exit, or fallback
        raise SystemExit(f"Failed to initialize ChromaDB with OpenAI: {e}")

else:
    # Handle the case where OpenAI key is not found
    logging.error("OPENAI_API_KEY environment variable not set.")
    # Exit or raise error because the current setup relies on OpenAI embeddings
    raise SystemExit("OPENAI_API_KEY environment variable not set.")


# --- Flask App Setup ---
# Serve static files (index.html, style.css, script.js) from the root project directory
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Function to Setup Vector Database (Load Data) ---
# This function mainly checks if the collection seems populated
def setup_vector_db():
    """Checks if the ChromaDB collection is populated."""
    global collection # Use the globally defined collection object
    try:
        if collection is None:
            logging.error("ChromaDB collection object is not initialized.")
            # Attempt to re-initialize (optional, might hide earlier issues)
            # collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
            # if collection is None:
            #    raise SystemExit("Failed to get collection during setup check.")
            return # Exit if collection isn't ready

        count = collection.count()
        if count > 0:
            logging.info(f"Collection '{COLLECTION_NAME}' already populated ({count} items).")
            # You might consider not embedding again if count > 0, as per original logs
            # If you have a separate script for populating, this check is sufficient.
            # If this app is supposed to populate, add the logic here based on count==0.
        else:
            logging.warning(f"Collection '{COLLECTION_NAME}' is empty. Loading data from reviews file...")
            
            # Load the reviews data from JSON file
            json_file_path = 'rag_backend/dustin_photography_reviews_cleaned.json'
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    reviews_data = json.load(f)
                    
                logging.info(f"Loaded {len(reviews_data)} reviews from {json_file_path}")
                
                # Prepare data for ChromaDB
                documents = []  # The text content to be embedded
                metadatas = []  # Associated metadata for each document
                ids = []        # Unique IDs for each document
                
                for i, review in enumerate(reviews_data):
                    # Skip entries with no content_text
                    if "content_text" not in review or not review["content_text"].strip():
                        continue
                        
                    # Add the review text to documents
                    documents.append(review.get("content_text", ""))
                    
                    # Create metadata from available fields
                    metadata = {
                        "title": review.get("title", ""),
                        "url": review.get("url", ""),
                        "date": review.get("date", "")
                    }
                    metadatas.append(metadata)
                    
                    # Create a unique ID
                    ids.append(f"review_{i}")
                
                # Add the documents to the collection
                if documents:
                    batch_size = 100  # Process in batches to avoid potential issues with large datasets
                    for i in range(0, len(documents), batch_size):
                        end_idx = min(i + batch_size, len(documents))
                        collection.add(
                            documents=documents[i:end_idx],
                            metadatas=metadatas[i:end_idx],
                            ids=ids[i:end_idx]
                        )
                        logging.info(f"Added batch {i//batch_size + 1} ({i} to {end_idx}) to ChromaDB collection")
                    
                    logging.info(f"Successfully added {len(documents)} reviews to ChromaDB collection")
                else:
                    logging.warning("No valid documents found to add to ChromaDB collection")
            
            except Exception as e:
                logging.error(f"Error loading or processing reviews data: {e}")

    except Exception as e:
        logging.error(f"Error accessing collection '{COLLECTION_NAME}' during setup check: {e}")
        # Depending on the error, you might want to raise it or handle differently


# --- LLM Response Generation ---
def generate_gpt_response(user_query, context_docs, metadata=None):
    """
    Generate a conversational response using OpenAI's GPT model based on retrieved context.
    
    Args:
        user_query: The user's question
        context_docs: List of relevant document contents retrieved from the vector DB
        metadata: Optional list of metadata for each document
    
    Returns:
        A conversational response from GPT
    """
    try:
        # Create a system prompt that explains the assistant's role
        system_prompt = """
        You are a helpful and knowledgeable photography equipment expert who provides information based on Photography.
        Answer the user's question based on the provided review excerpts only.
        Keep your answers concise, informative and conversational.
        If the provided context doesn't contain relevant information to answer the question, politely say so.
        Don't make up information or reference reviews that aren't in the provided context.
        """
        
        # Build the context from the retrieved documents - Fixed the 'meta' not defined error
        context_items = []
        for i, doc in enumerate(context_docs):
            if metadata and i < len(metadata):
                meta_info = metadata[i]  # Get the metadata for this document
                context_string = f"Review: {doc} (Title: {meta_info.get('title', 'Unknown')}, Date: {meta_info.get('date', 'Unknown')})"
                context_items.append(context_string)
                # Log each document and its metadata for debugging
                logging.info(f"RAG Context [{i+1}]: Title: {meta_info.get('title', 'Unknown')}")
                # Only show preview of doc content to avoid overwhelming logs
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                logging.info(f"RAG Content Preview [{i+1}]: {preview}")
            else:
                context_items.append(f"Review: {doc}")
        
        context = "\n\n---\n\n".join(context_items)
        
        # Create the user message including the query and context
        user_message = f"Question: {user_query}\n\nContext from reviews:\n{context}"
        
        # Log the FULL prompt that's being sent to GPT
        logging.info("=" * 80)
        logging.info("COMPLETE PROMPT SENT TO GPT:")
        logging.info("-" * 40)
        logging.info(f"SYSTEM PROMPT:\n{system_prompt}")
        logging.info("-" * 40)
        logging.info(f"USER PROMPT:\n{user_message}")
        logging.info("=" * 80)
        
        # Log that we're sending the query to GPT
        logging.info(f"Sending query to GPT with {len(context_docs)} context documents")
        
        # Generate the response using GPT
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Can use "gpt-4" for potentially better responses if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Balanced between creativity and accuracy
            max_tokens=500    # Keep responses reasonably sized
        )
        
        # Extract and return the response text
        if response.choices and len(response.choices) > 0:
            response_text = response.choices[0].message.content
            # Log the GPT response
            logging.info("=" * 80)
            logging.info("GPT RESPONSE:")
            logging.info(response_text)
            logging.info("=" * 80)
            return response_text
        else:
            return "Sorry, I couldn't generate a response for your query."
    
    except Exception as e:
        logging.error(f"Error generating GPT response: {e}")
        return f"Sorry, I encountered an error while processing your query. Please try again."


def rerank_documents(query, documents, metadata=None):
    """
    Reranks retrieved documents using a cross-encoder model for better relevancy.
    
    Args:
        query: The user query
        documents: List of documents retrieved from the initial vector search
        metadata: Optional list of metadata for each document
    
    Returns:
        Tuple of (reranked_documents, reranked_metadata, scores)
    """
    if not documents:
        return [], [], []
    
    # Load the reranker model if not already loaded
    model = load_reranker()
    if model is None:
        logging.warning("Reranker model could not be loaded, using original ranking")
        return documents, metadata or [], [0] * len(documents)
    
    # Create document pairs for scoring
    pairs = [(query, doc) for doc in documents]
    
    try:
        # Get scores from reranker model
        logging.info(f"Reranking {len(documents)} documents")
        scores = model.predict(pairs)
        
        # Create list of (doc, meta, score) tuples for sorting
        if metadata:
            items = list(zip(documents, metadata, scores))
        else:
            items = list(zip(documents, [{} for _ in documents], scores))
        
        # Sort by score in descending order
        sorted_items = sorted(items, key=lambda x: x[2], reverse=True)
        
        # Unpack the sorted results
        sorted_docs, sorted_meta, sorted_scores = zip(*sorted_items)
        
        # Log reranking results
        for i, (score, doc) in enumerate(zip(sorted_scores, sorted_docs)):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            logging.info(f"Reranked doc {i+1}: Score={score:.4f}, Preview={preview}")
            
        return list(sorted_docs), list(sorted_meta), list(sorted_scores)
        
    except Exception as e:
        logging.error(f"Error during document reranking: {e}")
        return documents, metadata or [], [0] * len(documents)

def generate_grounded_response(user_query, context_docs, metadata=None):
    """
    Generate a conversational response using a two-step process:
    1. Initial grounded response with GPT-3.5 Turbo for efficiency
    2. Refinement with GPT-4 for better quality and natural tone
    
    Args:
        user_query: The user's question
        context_docs: List of relevant document contents retrieved from the vector DB
        metadata: Optional list of metadata for each document
    
    Returns:
        A refined conversational response that is grounded in the provided context
    """
    try:
        # Create a system prompt that enforces grounding but avoids explicit source attribution
        system_prompt_initial = """
        You are a knowledgeable photography expert who provides information about photography equipment.
        
        RESPONSE RULES:
        1. Only use facts found in the provided review excerpts
        2. Do not mention sources or say phrases like "according to reviews"
        3. If the reviews don't contain information needed to answer, simply say you don't have information about that specific topic
        4. Do not make up information not present in the reviews
        5. Be concise but complete in your response
        
        Format your response as if you're stating facts that you personally know.
        """
        
        # Build the context from the retrieved documents with hidden source indicators
        context_items = []
        for i, doc in enumerate(context_docs):
            source_id = f"__SOURCE_{i+1}__"  # Hidden marker for internal reference only
            if metadata and i < len(metadata):
                meta_info = metadata[i]
                title = meta_info.get('title', 'Unknown Review')
                date = meta_info.get('date', 'Unknown Date')
                url = meta_info.get('url', '')
                
                # Format source with metadata
                context_string = f"{source_id} TITLE: {title} | DATE: {date} | CONTENT: {doc}"
                context_items.append(context_string)
                
                # Log each document and its metadata for debugging
                logging.info(f"Grounded RAG Context {i+1}: {title} ({date})")
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                logging.info(f"Content Preview {i+1}: {preview}")
            else:
                context_string = f"{source_id} CONTENT: {doc}"
                context_items.append(context_string)
        
        # Join all context items with clear separators
        context = "\n\n" + "\n\n".join(context_items)
        
        # Create a user message for the initial response
        user_message_initial = f"""
        USER QUESTION: {user_query}
        
        REVIEWS TO BASE YOUR ANSWER ON:
        {context}
        
        Remember: Do not reference the sources directly and only include information found in these reviews.
        """
        
        # Log the initial prompt
        logging.info("=" * 80)
        logging.info("INITIAL GPT-3.5 GROUNDED PROMPT:")
        logging.info(f"Query: {user_query}")
        logging.info("=" * 80)
        
        # Generate the initial response using GPT-3.5 Turbo
        initial_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo for initial grounding (more cost-effective)
            messages=[
                {"role": "system", "content": system_prompt_initial},
                {"role": "user", "content": user_message_initial}
            ],
            temperature=0.3,  # Lower temperature for factual accuracy
            max_tokens=800    # Allow space for comprehensive response
        )
        
        # Extract the initial response
        if initial_response.choices and len(initial_response.choices) > 0:
            raw_initial_response = initial_response.choices[0].message.content
            logging.info("INITIAL GPT-3.5 RESPONSE:")
            logging.info(raw_initial_response)
            
            # Now create a refinement prompt for GPT-4
            system_prompt_refine = """
            You are a helpful and knowledgeable photography expert who provides friendly, conversational responses.
            
            Your job is to refine the provided response:
            1. Make the tone more natural, conversational and authoritative
            2. Remove any remaining source attributions or phrases like "according to reviews"
            3. Format the answer in a clear, readable way
            4. Ensure the response sounds like it's coming from an expert who personally knows about photography
            5. Do NOT add any new information - only refine the existing content
            6. Keep the response concise and to the point
            """
            
            user_message_refine = f"""
            ORIGINAL QUESTION: {user_query}
            
            INITIAL RESPONSE TO REFINE:
            {raw_initial_response}
            
            Please refine this response to be more natural and conversational while maintaining factual accuracy.
            Do NOT add any new information beyond what's in the initial response. Never mention "reviews" or sources.
            """
            
            # Generate the refined response with GPT-4
            refined_response = openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for refinement
                messages=[
                    {"role": "system", "content": system_prompt_refine},
                    {"role": "user", "content": user_message_refine}
                ],
                temperature=0.5,  # Slightly higher temperature for natural language
                max_tokens=800    # Allow space for the refined response
            )
            
            if refined_response.choices and len(refined_response.choices) > 0:
                final_response = refined_response.choices[0].message.content
                
                # Post-process to remove any remaining source references
                final_response = re.sub(r'\[SOURCE \d+\]', '', final_response)
                final_response = re.sub(r'\(SOURCE \d+\)', '', final_response)
                final_response = re.sub(r'according to (?:the )?reviews', '', final_response, flags=re.IGNORECASE)
                final_response = re.sub(r'the reviews mention(?:d)?', '', final_response, flags=re.IGNORECASE)
                final_response = re.sub(r'based on (?:the )?reviews', '', final_response, flags=re.IGNORECASE)
                final_response = re.sub(r'as mentioned in (?:the )?reviews', '', final_response, flags=re.IGNORECASE)
                final_response = re.sub(r' +', ' ', final_response)  # Clean up spaces
                final_response = re.sub(r' ,', ',', final_response)
                final_response = re.sub(r' \.', '.', final_response)
                
                # Log the final refined response
                logging.info("=" * 80)
                logging.info("FINAL REFINED GPT-4 RESPONSE:")
                logging.info(final_response)
                logging.info("=" * 80)
                
                return final_response
            else:
                # If refinement fails, return the initial response after cleaning
                cleaned_initial = re.sub(r'\[SOURCE \d+\]', '', raw_initial_response)
                cleaned_initial = re.sub(r'according to (?:the )?reviews', '', cleaned_initial, flags=re.IGNORECASE)
                return cleaned_initial
        else:
            return "Sorry, I couldn't generate a response for your query."
    
    except Exception as e:
        logging.error(f"Error generating two-step grounded response: {e}")
        return f"Sorry, I encountered an error while processing your query. Please try again."

# --- Flask Routes ---

# Route to serve the main HTML page
@app.route('/')
def serve_index():
    """Serves the index.html file."""
    logging.info("Serving index.html")
    # send_from_directory looks inside the static_folder defined for the app
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint to handle user queries
@app.route('/query', methods=['POST'])
def handle_query():
    """Handles user queries by searching the vector database and generating responses with GPT."""
    global collection # Ensure access to the global collection object
    if collection is None:
        logging.error("Query attempted but collection is not available.")
        return jsonify({"error": "Vector database not ready"}), 503 # Service Unavailable

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Missing 'query' in request data"}), 400

    logging.info(f"Received user query: '{user_query}'")
    logging.info("=" * 50)  # Add separator for better readability in logs

    try:
        # Initial vector search query
        logging.info(f"Searching vector database for: '{user_query}'")
        results = collection.query(
            query_texts=[user_query],
            n_results=10,  # Retrieve more candidates for reranking
            include=['documents', 'metadatas', 'distances'] # Include relevant data
        )

        # Extract context documents and metadata
        context_docs = results.get('documents', [[]])[0]
        metadata_list = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        # Log initial retrieval results
        logging.info(f"Initially retrieved {len(context_docs)} documents from vector database")
        if distances:
            for i, distance in enumerate(distances):
                logging.info(f"Initial document {i+1} similarity score: {1-distance:.4f}")  # Convert distance to similarity

        if not context_docs:
            response_text = "Sorry, I couldn't find any relevant information about that in my knowledge base of photography equipment reviews."
            logging.info("No relevant documents found in the vector database")
        else:
            # Apply reranking to improve document relevancy
            logging.info("Applying reranking to improve document relevance...")
            reranked_docs, reranked_metadata, reranker_scores = rerank_documents(user_query, context_docs, metadata_list)
            
            # Use only top 5 documents after reranking for better focus
            top_k = min(5, len(reranked_docs))
            selected_docs = reranked_docs[:top_k]
            selected_metadata = reranked_metadata[:top_k]
            
            # Generate a grounded response using the reranked documents
            logging.info(f"Generating grounded response using top {top_k} reranked documents")
            response_text = generate_grounded_response(user_query, selected_docs, selected_metadata)

        logging.info(f"Final response: '{response_text}'")
        logging.info("=" * 50)  # Add separator for better readability in logs
        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"Error during query processing for query '{user_query}': {e}")
        return jsonify({"error": "Failed to process query due to an internal error"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Check DB setup *after* potential initialization but before running the app
    setup_vector_db()

    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your network
    # debug=True enables auto-reloading and detailed error pages (turn off in production)
    logging.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

