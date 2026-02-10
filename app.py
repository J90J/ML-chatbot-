import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Page Config
st.set_page_config(page_title="ML Textbook Chatbot", page_icon="📚")
st.title("📚 Machine Learning Textbook Chatbot")

# Sidebar for API Keys configuration
# Check if keys are available in secrets (for deployment)
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Local fallback or manual entry
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "PINECONE_API_KEY" in st.secrets:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
else:
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()
if not pinecone_api_key:
    st.warning("Please enter your Pinecone API Key to proceed.")
    st.stop()

# Initialize Clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index_name = "machine-learning-textbook"
index = pc.Index(index_name)

# Embeddings Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Session State
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to Retrieve Context
def retrieve_context(query, top_k=5, namespace="ns500"):
    try:
        query_embedding = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        contexts = [match['metadata']['text'] for match in results['matches']]
        return "\n\n".join(contexts)
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return ""

# Chat Input
if prompt := st.chat_input("Ask a question about Machine Learning..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        # 1. Retrieve Context
        context = retrieve_context(prompt)
        
        # 2. Prepare Prompt with Context (RAG)
        system_prompt = f"""You are a helpful assistant for a Machine Learning textbook. 
Use the following context to answer the user's question. 
If the answer is not in the context, say you don't know based on the textbook.

Context:
{context}
"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        # Append only recent history to avoid context limit issues (optional optimization)
        messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
