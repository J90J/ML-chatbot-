import json
import os

NOTEBOOK_PATH = "/Users/jensjung/Desktop/Mini Project Part 2/Task description /Mini Project 2 Part 1 and 2.ipynb"

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Helper to find cell by ID or content (if ID missing)
    def find_cell_index(cell_id):
        for i, cell in enumerate(nb['cells']):
            if cell.get('metadata', {}).get('id') == cell_id:
                return i
        return -1

    # --- Task 1: Load PDF & Chunking ---
    idx = find_cell_index('pyUUcnqGw39B')
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "from langchain_community.document_loaders import PyMuPDFLoader\n",
            "\n",
            "loader = PyMuPDFLoader(\"machine-learning.pdf\")\n",
            "docs = loader.load()\n",
            "\n",
            "page_texts = [doc.page_content for doc in docs]\n",
            "page_numbers = [doc.metadata.get('page', 0) + 1 for doc in docs]\n",
            "\n",
            "print(f\"Loaded {len(docs)} pages.\")"
        ]

    idx = find_cell_index('lmeX3Zx7w39B') # Chunking
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
            "\n",
            "chunk_size = 500\n",
            "overlap = 50\n",
            "\n",
            "text_splitter = RecursiveCharacterTextSplitter(\n",
            "    chunk_size=chunk_size,\n",
            "    chunk_overlap=overlap,\n",
            "    length_function=len,\n",
            "    is_separator_regex=False\n",
            ")\n",
            "\n",
            "# Split the documents directly from the loaded docs\n",
            "# This handles the overlap and page metadata automatically\n",
            "chunks = text_splitter.split_documents(docs)\n",
            "\n",
            "print(f\"Generated {len(chunks)} chunks with size {chunk_size}.\")\n",
            "\n",
            "# For manual DataFrame construction in next step:\n",
            "chunked_texts = [chunk.page_content for chunk in chunks]\n",
            "chunk_page_numbers = [chunk.metadata.get('page', 0) + 1 for chunk in chunks]"
        ]

    # --- Task 2: Data Prep & Embeddings ---
    idx = find_cell_index('bUFWi8B8w39B') # OpenAI Key
    if idx != -1:
         nb['cells'][idx]['source'] = [
            "# Setup OpenAI API Key\n",
            "import os\n",
            "from langchain_openai import OpenAIEmbeddings\n",
            "from openai import OpenAI\n",
            "\n",
            "os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY_HERE' # Replace with your key\n",
            "client = OpenAI()\n",
            "\n",
            "def get_embedding(text, model=\"text-embedding-3-small\"):\n",
            "   text = text.replace(\"\\n\", \" \")\n",
            "   return client.embeddings.create(input = [text], model=model).data[0].embedding"
        ]
    
    # DataFrame creation (No ID, it's the cell after bUFWi8B8w39B)
    # We can iterate to find it or just assume index (risky if NB changes).
    # Let's search by content "TODO: Write your code here"
    for i, cell in enumerate(nb['cells']):
        if "TODO: Write your code here" in "".join(cell['source']) and "Convert the list of texts into a DataFrame" in "".join(cell['source']):
            nb['cells'][i]['source'] = [
                "import pandas as pd\n",
                "\n",
                "# Convert to DataFrame\n",
                "df = pd.DataFrame({\n",
                "    'text': chunked_texts,\n",
                "    'page_number': chunk_page_numbers\n",
                "})\n",
                "\n",
                "# Preprocess: Clean newlines/punctuation (basic)\n",
                "df['text'] = df['text'].apply(lambda x: x.replace('\\n', ' ').strip())\n",
                "\n",
                "# Generate Embeddings\n",
                "print(\"Generating embeddings...\")\n",
                "# Using batch processing via Langchain for efficiency, or the manual function\n",
                "# Let's use the provided function for compliance\n",
                "df['embeddings'] = df['text'].apply(lambda x: get_embedding(x))\n",
                "\n",
                "print(f\"Generated embeddings for {len(df)} chunks.\")\n",
                "df.head()"
            ]
            break

    # --- Task 3: Pinecone Indexing ---
    idx = find_cell_index('fGpeUS20w39C')
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "from pinecone import Pinecone, ServerlessSpec\n",
            "import time\n",
            "\n",
            "PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'YOUR_PINECONE_KEY_HERE')\n",
            "INDEX_NAME = \"machine-learning-textbook\"\n",
            "\n",
            "pc = Pinecone(api_key=PINECONE_API_KEY)\n",
            "\n",
            "# Create Index if not exists\n",
            "existing_indexes = [index.name for index in pc.list_indexes()]\n",
            "if INDEX_NAME not in existing_indexes:\n",
            "    pc.create_index(\n",
            "        name=INDEX_NAME,\n",
            "        dimension=1536, # text-embedding-3-small dim\n",
            "        metric=\"cosine\",\n",
            "        spec=ServerlessSpec(cloud=\"aws\", region=\"us-east-1\")\n",
            "    )\n",
            "    while not pc.describe_index(INDEX_NAME).status['ready']:\n",
            "        time.sleep(1)\n",
            "\n",
            "index = pc.Index(INDEX_NAME)\n",
            "print(f\"Connected to index: {INDEX_NAME}\")\n",
            "\n",
            "# Upsert Embeddings\n",
            "namespace = \"ns500\"\n",
            "vectors_to_upsert = []\n",
            "for i, row in df.iterrows():\n",
            "    vectors_to_upsert.append({\n",
            "        \"id\": f\"vec_{i}\",\n",
            "        \"values\": row['embeddings'],\n",
            "        \"metadata\": {\n",
            "            \"text\": row['text'],\n",
            "            \"page_number\": row['page_number']\n",
            "        }\n",
            "    })\n",
            "    # Batch upsert every 100\n",
            "    if len(vectors_to_upsert) >= 100:\n",
            "        index.upsert(vectors=vectors_to_upsert, namespace=namespace)\n",
            "        vectors_to_upsert = []\n",
            "\n",
            "# Final batch\n",
            "if vectors_to_upsert:\n",
            "    index.upsert(vectors=vectors_to_upsert, namespace=namespace)\n",
            "\n",
            "print(f\"Upserted {len(df)} vectors to namespace {namespace}.\")"
        ]

    # --- Task 4: Query ---
    idx = find_cell_index('CeUBbCWNw39C') # Vector Store Init
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "from langchain_pinecone import PineconeVectorStore\n",
            "\n",
            "text_field = \"text\"\n",
            "vectorstore = PineconeVectorStore(\n",
            "    index,\n",
            "    OpenAIEmbeddings(model=\"text-embedding-3-small\"),\n",
            "    text_field,\n",
            "    namespace=\"ns500\" # Default to 500 for now\n",
            ")\n"
        ]

    idx = find_cell_index('JxXyjKosw39D') # Query Function
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "def query_pinecone_vector_store(query: str, top_k: int = 5, namespace: str = \"ns500\"):\n",
            "    # We can use the vectorstore directly or the index directly.\n",
            "    # Using vectorstore for Langchain integration is easier for retrieval\n",
            "    # But for raw query as requested:\n",
            "    query_embedding = get_embedding(query)\n",
            "    results = index.query(\n",
            "        namespace=namespace,\n",
            "        vector=query_embedding,\n",
            "        top_k=top_k,\n",
            "        include_values=False,\n",
            "        include_metadata=True\n",
            "    )\n",
            "    return results\n"
        ]

    # --- Task 5: RAG Generation ---
    idx = find_cell_index('CC2NzvfOw39D') # Answer Query
    if idx != -1:
        nb['cells'][idx]['source'] = [
            "def ask_with_rag(query, namespace=\"ns500\"):\n",
            "    # 1. Retrieve\n",
            "    results = query_pinecone_vector_store(query, top_k=5, namespace=namespace)\n",
            "    \n",
            "    # 2. Construct Context\n",
            "    contexts = [match['metadata']['text'] for match in results['matches']]\n",
            "    context_str = \"\\n\\n\".join(contexts)\n",
            "    \n",
            "    # 3. Augment Prompt\n",
            "    system_prompt = f\"\"\"You are a helpful assistant. Use the following context to answer the user's question.\n",
            "If the answer is not in the context, say so.\n",
            "\n",
            "Context:\n",
            "{context_str}\n",
            "\"\"\"\n",
            "    \n",
            "    # 4. Generate\n",
            "    response = client.chat.completions.create(\n",
            "        model=\"gpt-3.5-turbo\",\n",
            "        messages=[\n",
            "            {\"role\": \"system\", \"content\": system_prompt},\n",
            "            {\"role\": \"user\", \"content\": query}\n",
            "        ]\n",
            "    )\n",
            "    return response.choices[0].message.content\n"
        ]
        
    # Test Prompts (No ID, it's the cell after CC2NzvfOw39D)
    # Search for "Create 5 test prompts"
    for i, cell in enumerate(nb['cells']):
        if "Create 5 test prompts" in "".join(cell['source']):
            nb['cells'][i]['source'] = [
                "test_queries = [\n",
                "    \"What is supervised learning?\",\n",
                "    \"Explain the bias-variance tradeoff.\",\n",
                "    \"How does the decision tree algorithm work?\",\n",
                "    \"How to cook an egg?\", # Irrelevant\n",
                "    \"Who won the 1994 World Cup?\" # Irrelevant\n",
                "]\n",
                "\n",
                "print(\"--- RAG Testing ---\")\n",
                "for q in test_queries:\n",
                "    print(f\"\\nQuery: {q}\")\n",
                "    answer = ask_with_rag(q)\n",
                "    print(f\"Answer: {answer}\")\n"
            ]
            break

    
    # Save
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook()
