import os
import numpy as np
from google import genai
#from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st

# --- CONFIGURATION --- #

# Load environment variables. Ensure GEMINI_API_KEY is set.
#load_dotenv(r"C:\eaton_rag_app\.env")

# 1. LLM Client: Initialize the Gemini Client
@st.cache_resource
def get_gemini_client():
    try:
        # st.secrets is the secure way to access the key on Streamlit Cloud
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except (AttributeError, KeyError):
        # This fallback is for running *locally* if you still want to use .env
        gemini_key = os.getenv("GEMINI_API_KEY") 

    if not gemini_key:
        st.error("FATAL ERROR: GEMINI_API_KEY not found.")
        # Do not proceed without the key
        st.stop()

    return genai.Client(api_key=gemini_key) 

client = get_gemini_client()

# 2. Embedding Model: Initialize the local Sentence Transformer model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model = get_embedding_model()

# 3. LLM Model Settings
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 250
SYSTEM_PROMPT = (
    "You are a knowledgeable and professional assistant. "
    "You provide accurate, helpful, and concise information about Eaton, "
    "including company overview, products, business sectors, and sustainability. "
    "Always base your answers on the provided knowledge base and maintain a polite, professional tone."
)

# ------------------ KNOWLEDGE BASE SETUP ------------------ #

# Use Streamlit caching to load and process knowledge only once
@st.cache_data
def process_knowledge(file_path="eaton_knowledge.txt", chunk_size=200):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            knowledge = f.read()
    except FileNotFoundError:
        st.error("Knowledge file 'eaton_knowledge.txt' not found.")
        return [], None
    
    # Split knowledge
    words = knowledge.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # Generate and cache embeddings
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    return chunks, chunk_embeddings

KNOWLEDGE_CHUNKS, CHUNK_EMBEDDINGS = process_knowledge()


# ------------------ CHAT FUNCTIONS ------------------ #

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)


def chat_with_rag(user_input):
    messages = st.session_state.messages
    
    # RAG Logic
    question_embedding = embedding_model.encode(user_input, convert_to_numpy=True)
    similarities = [cosine_similarity(question_embedding, chunk_emb) for chunk_emb in CHUNK_EMBEDDINGS]
    top_indices = np.argsort(similarities)[-2:]
    relevant_chunks = "\n\n".join([KNOWLEDGE_CHUNKS[i] for i in reversed(top_indices)])

    # Format Context for the current turn
    contextual_user_prompt = f"Eaton Knowledge Base (Relevant):\n{relevant_chunks}\n\nUser Question: {user_input}"
    
    # Prepare contents for the Gemini API call
    gemini_contents = []
    # Add history from session state (skipping the system prompt at index 0)
    for msg in messages[1:]:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_contents.append({'role': role, 'parts': [{'text': msg['content']}]})

    # Add the current RAG-augmented user prompt
    gemini_contents.append({'role': 'user', 'parts': [{'text': contextual_user_prompt}]})
    
    # LLM Call
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=gemini_contents,
            config={
                "system_instruction": SYSTEM_PROMPT, 
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_TOKENS,
            }
        )
    except Exception as e:
        return f"[Error: Gemini API Call Failed] Ensure your GEMINI_API_KEY is correct and set in your environment. Details: {e}"

    # Get assistant reply, add to history
    reply = response.text
    
    # Add original user input and the reply to the session history for display
    # This ensures both messages are saved for the next full script rerun (history display loop)
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": reply})

    return reply

# ------------------ STREAMLIT UI IMPLEMENTATION ------------------ #

st.set_page_config(page_title="Eaton RAG Chatbot (Gemini + Streamlit)", layout="wide")
st.title("⚡️ Eaton RAG Chatbot")
st.caption(f"Powered by **{MODEL}** (via Gemini API) and **Sentence-Transformers** (Local Embeddings)")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Display conversation history (This loop runs first)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Eaton's products, sectors, or sustainability..."):
    
    # 1. DISPLAY CURRENT USER PROMPT IMMEDIATELY (The Fix!)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Generate and display assistant response
    # The chat_with_rag function is now responsible for generating the response
    # AND saving both the user and assistant messages to st.session_state.messages
    with st.chat_message("assistant"):
        with st.spinner(f"Asking {MODEL}..."):
            response = chat_with_rag(prompt)
            st.markdown(response)

# Display system metrics in the sidebar
st.sidebar.header("System Metrics")
st.sidebar.metric("Max Response Length", f"{MAX_TOKENS} tokens") 
st.sidebar.markdown("---")
st.sidebar.markdown("**Setup Status**")
if CHUNK_EMBEDDINGS is not None and len(KNOWLEDGE_CHUNKS) > 0:
    st.sidebar.success(f"Knowledge Loaded: {len(KNOWLEDGE_CHUNKS)} chunks")
else:
    st.sidebar.warning("Knowledge Base not loaded.")
