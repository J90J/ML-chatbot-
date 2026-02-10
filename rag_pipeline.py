import os
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    # If langchain_community is not installed or different version
    # Fallback or allow error to propagate after pip install
    from langchain.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from typing import List, Dict

# Configuration
PDF_PATH = "/Users/jensjung/Desktop/Mini Project Part 2/Task description /machine-learning.pdf"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
CHUNK_SIZES = [500, 1000, 2500]
OVERLAP = 50

# Set API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_pdf(file_path: str):
    """Loads PDF and extracts text/metadata per page."""
    print(f"Loading PDF from {file_path}...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} pages.")
    return docs

def process_chunks(docs, chunk_size, overlap):
    """Splits documents into chunks of a specific size."""
    print(f"Splitting text with Chunk Size: {chunk_size}, Overlap: {overlap}...")
    
    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def generate_embeddings_df(chunks):
    """Clean text and generate embeddings."""
    print("Generating embeddings (this may take a moment)...")
    
    # Initialize OpenAI Embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    data = []
    for chunk in chunks:
        # Preprocess text: Remove newlines and excess whitespace
        clean_text = chunk.page_content.replace("\n", " ").strip()
        
        # We process one by one or batch? Batch is better but let's stick to simple loop for logic clarity
        # actually Langchain handles this well, but let's do manual for DF creation as requested in one of the tasks
        
        data.append({
            "text": clean_text,
            "page_number": chunk.metadata.get("page", -1) + 1, # 1-based indexing for user friendliness
            "chunk_size": len(clean_text)
        })
        
    df = pd.DataFrame(data)
    
    # Generate embeddings in batch
    texts = df["text"].tolist()
    embeddings = embedding_model.embed_documents(texts)
    df["embeddings"] = embeddings
    
    return df

def run_pipeline():
    # Task 1: Load PDF
    raw_docs = load_pdf(PDF_PATH)
    
    results = {}
    
    for size in CHUNK_SIZES:
        print(f"\n--- Processing for Chunk Size {size} ---")
        
        # Task 1 (cont): Chunking
        chunks = process_chunks(raw_docs, size, OVERLAP)
        
        # Task 2: Data Prep & Embeddings
        # Note: In a real "Improvement" scenario, we might only generate embeddings for the 'best' size, 
        # but to compare them we need to index them all.
        df = generate_embeddings_df(chunks)
        results[size] = df
        
        print(f"Completed processing for size {size}. Embeddings generated: {len(df)}")
        print(df.head(2))

    # Task 3 & 4: Indexing & Experimentation
    from pinecone import Pinecone, ServerlessSpec
    import time
    
    PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_KEY")
    INDEX_NAME = "machine-learning-textbook"
    
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # Create Index
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    
    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
    
    experiment_results = {}
    test_query = "What is supervised learning?"
    query_vec = OpenAIEmbeddings(model="text-embedding-3-small").embed_query(test_query)

    for size in CHUNK_SIZES:
        print(f"\n--- Processing for Chunk Size {size} ---")
        
        # Chunking
        chunks = process_chunks(raw_docs, size, OVERLAP)
        # Embeddings
        df = generate_embeddings_df(chunks)
        
        # Upsert to Pinecone
        namespace = f"ns{size}"
        print(f"Upserting to namespace {namespace}...")
        vectors = []
        for i, row in df.iterrows():
            vectors.append({
                "id": f"vec_{i}",
                "values": row['embeddings'],
                "metadata": {"text": row['text'], "page": row['page_number']}
            })
            
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            
        print(f"Upsert complete for {namespace}.")
        
        # Query (Task 4)
        print(f"Querying {namespace}...")
        results = index.query(
            namespace=namespace,
            vector=query_vec,
            top_k=3,
            include_metadata=True
        )
        
        experiment_results[size] = [m['metadata']['text'][:200] + "..." for m in results['matches']]
        print("Top Result Snippet:", experiment_results[size][0])

    print("\n--- Experimentation Results Summary ---")
    print(experiment_results)
    return experiment_results

if __name__ == "__main__":
    run_pipeline()
