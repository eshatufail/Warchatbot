import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# ====================== CONFIGURATION ======================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not set. Please add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

ARTICLE_URL = "https://theweek.com/92967/are-we-heading-towards-world-war-3"

# ====================== PAGE SETUP ======================
st.set_page_config(
    page_title="World War 3 Analyst",
    page_icon="🌍",
    layout="wide"
)

# Sidebar - Theme & Controls
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

# Professional Custom CSS
st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .stChatMessage {{ 
        border-radius: 12px; 
        border: 1px solid #444; 
        margin-bottom: 15px; 
        padding: 12px;
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
@st.cache_resource(show_spinner="Loading global intelligence data...")
def setup_knowledge_base():
    try:
        loader = WebBaseLoader(ARTICLE_URL)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(data)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma.from_documents(chunks, embeddings)
        return vectorstore.as_retriever()
        
    except Exception as e:
        st.error(f"❌ Failed to load knowledge base: {str(e)}")
        return None

if st.session_state.retriever is None:
    st.session_state.retriever = setup_knowledge_base()

# ====================== CHAT INTERFACE ======================
# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Ask anything about World War 3 or Geopolitics..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    context_docs = st.session_state.retriever.invoke(prompt) if st.session_state.retriever else []
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    system_prompt = f"""
    You are a highly professional Geopolitical Analyst.
    Use the context below when answering about World War 3:
    {context_text}

    - Answer in a serious, analytical tone.
    - For greetings or general questions, respond naturally.
    - Keep responses clear and insightful.
    """

    messages = [("system", system_prompt)]
    for msg in st.session_state.chat_history[-8:]:
        role = "human" if isinstance(msg, HumanMessage) else "assistant"
        messages.append((role, msg.content))
    messages.append(("human", prompt))

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing geopolitical situation..."):
            try:
                response = llm.invoke(messages)
                st.markdown(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
