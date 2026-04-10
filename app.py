import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# ====================== CONFIGURATION ======================
load_dotenv()  # Load environment variables

# Your new Groq API Key (safely added)
GROQ_API_KEY = "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"

ARTICLE_URL = "https://theweek.com/92967/are-we-heading-towards-world-war-3"

# ====================== PAGE SETUP ======================
st.set_page_config(
    page_title="World War 3 Analyst",
    page_icon="🌍",
    layout="wide"
)

# Theme Selection
with st.sidebar:
    st.title("🎨 Appearance")
    theme_mode = st.radio("Choose Theme:", ["Dark", "Light"])
    
    if theme_mode == "Dark":
        bg_color = "#0e1117"
        text_color = "white"
    else:
        bg_color = "white"
        text_color = "black"

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Custom CSS for Professional Look
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .stChatMessage {{ 
        border-radius: 12px; 
        border: 1px solid #444; 
        margin-bottom: 12px; 
        padding: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("⚔️ World War 3 Intelligence Analyst")
st.caption("Real-time Geopolitical Analysis | Powered by Groq + RAG")

# ====================== SESSION STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ====================== KNOWLEDGE BASE (RAG) ======================
@st.cache_resource
def setup_knowledge_base():
    try:
        with st.spinner("Loading global intelligence data..."):
            loader = WebBaseLoader(ARTICLE_URL)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(data)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        return None

if st.session_state.retriever is None:
    st.session_state.retriever = setup_knowledge_base()

# ====================== CHAT INTERFACE ======================
# Display previous messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Ask anything about World War 3 or Geopolitics..."):
    
    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    # LLM Setup
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    # Retrieve relevant context
    context_docs = st.session_state.retriever.invoke(prompt) if st.session_state.retriever else []
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Professional System Prompt
    system_prompt = f"""
    You are a highly professional Geopolitical Analyst and Intelligence Expert.
    Use the following context from The Week article when answering questions about World War 3:
    {context_text}

    - Answer in a serious, analytical, and confident tone.
    - If the question is general or a greeting, respond naturally using your knowledge.
    - Keep responses clear, well-structured, and insightful.
    """

    # Prepare messages with history (last 6 messages for context)
    messages = [("system", system_prompt)]
    for msg in st.session_state.chat_history[-6:]:
        role = "human" if isinstance(msg, HumanMessage) else "assistant"
        messages.append((role, msg.content))
    messages.append(("human", prompt))

    # Get AI Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing geopolitical situation..."):
            response = llm.invoke(messages)
            st.markdown(response.content)
    
    # Save assistant response to history
    st.session_state.chat_history.append(AIMessage(content=response.content))