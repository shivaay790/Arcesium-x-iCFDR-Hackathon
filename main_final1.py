####################################### 1) importing required libraries #######################################################################

import torch
if not hasattr(torch, '__path__'):
    torch.__path__ = []


import streamlit as st
import google.generativeai as genai
import re
import os
import asyncio

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

####################################### 2) intitializations ####################################################################################


# 1) UI of APP
st.set_page_config(
    page_title="PsyAssist - Mental Health Support",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# Custom CSS for dark-blue-green theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0a192f;
        color: #e6f1ff;
    }
    .stButton button {
        background-color: #4b86b4 !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    .stButton button:hover {
        background-color: #2a4d69 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 PsyAssist - Mental Health Companion") # App layout
st.caption("A safe space for emotional support and mental health resources")



# Configuration
PSYCH_CONTENT_DIR = "psychological_resources/"

# 2) Initialize Models
@st.cache_resource  # caches the model to reduces time to reloading
def load_model():
    try:
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()
chat_model = model
sentiment_model = model 

# 3) Initialize Vector Store for RAG
@st.cache_resource(show_spinner=False)
def initialize_rag():
    faiss_index_path = "vectorstore/index.faiss"

    if os.path.exists(faiss_index_path):
        return FAISS.load_local("vectorstore", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'), allow_dangerous_deserialization=True)
    
    try:
        loader = DirectoryLoader(
            PSYCH_CONTENT_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs = loader.load()
        # Add metadata: filename, page
        for doc in docs:
            doc.metadata['source'] = doc.metadata.get('source', 'unknown')
        
        text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=5000,
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
        vectorstore.save_local("vectorstore")

        return vectorstore
    # FAISS is a vec serach database for semantic search
    except Exception as e:
        st.error(f"RAG Initialization Error: {str(e)}")
        return None

vector_store = initialize_rag()

# 4) Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm PsyAssist. How can I support you today?"}
    ]

# 5) Crisis detection and safety settings
CRISIS_TRIGGERS = [
    r'\b(suicidal|suicide|kill myself|want to die|end it all|no reason to live|take my own life)\b',
    r'\b(abused|being abused|sexual abuse|violence at home|domestic violence|raped|molested|assaulted)\b',
    r'\b(overdose|over dosing|cutting|self[-\s]?harm|hurt myself|burning myself|self mutilation)\b',
    r'\b(can[’\']?t go on|depressed|lost all hope|crying all day|hopeless|worthless)\b'
]

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]


########################################################  3) helper functions #################################################################

# 1) get context
def retrieve_psychological_context(query):
    if not vector_store:
        return ""
    try:
        results = vector_store.similarity_search(query, k=2)
        return "\n".join([f"- {doc.page_content[:300].strip()}..." for doc in results])
    except Exception as e:
        st.error(f"Context retrieval error: {str(e)}")
        return ""

# 2) understand sentiment 
def analyze_sentiment(text):
    prompt = """Analyze this message for emotional state. Respond ONLY with:
        - Primary emotion (neutral, anxious, depressed, angry, suicidal)
        - Intensity (1-5)
        - Suicidal intent (yes/no)
        Format: emotion|intensity|suicidal"""
    
    try:
        response = sentiment_model.generate_content(
            prompt + text,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"max_output_tokens": 50}
        )
        
        parts = response.text.strip().split('|')
        return parts if len(parts) == 3 else ["neutral", "1", "no"]
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return ["neutral", "1", "no"]


# 3) give a response
def generate_response(text, sentiment):
    context = retrieve_psychological_context(text)
    previous_history = st.session_state.get("messages", [])

    prompt = f"""
    Generate a friendly, emotionally supportive response following these rules:
    
    --- CONTEXT ---
    {context}

    --- CHAT HISTORY ---
    {previous_history}

    --- USER MESSAGE ---
    {text}

    --- REQUIREMENTS ---
    - Keep it brief (max 50 words)
    - strictly do not assume any pervious conversation with the user if not present in chat history
    - if you dont understand why does a user gives a response as for context
    - Use a natural, casual tone
    - Validate user's emotion empathetically
    - Suggest practical coping strategies
    - Do not hallucinate or give facts not in context/chat history
    - Reply in same language as user (detect automatically)
    - Avoid switching languages randomly
    - Ask a few questions before Offering diverse actionable suggestions
    """

    
    try:
        response = chat_model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"max_output_tokens": 200, "temperature": 1.2}
        )
        return response.text.strip()
    except Exception as e:
        return "I'm having trouble formulating a response right now. Please try again later."

# 4) check for crisis
def check_crisis(text, sentiment):
    if sentiment[2] == 'yes': 
        return True
    return any(re.search(pattern, text.lower()) for pattern in CRISIS_TRIGGERS)


########################################################  4) decleare & run functions #################################################################



# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):    # := is the walrus operator (assignment expression), so prompt gets the input
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process message
    with st.spinner("Processing..."):
        sentiment = analyze_sentiment(prompt)
        
        if check_crisis(prompt, sentiment):
            crisis_response = """
            <div class='crisis-alert'>
                🚨 CRISIS ALERT 🚨<br>
                It sounds like you're going through something serious. Please contact:<br>
                - National Suicide Prevention Lifeline: 988<br>
                - Crisis Text Line: Text HOME to 741741<br>
                - Your local emergency services
            </div>
            """
            st.session_state.messages.append({"role": "assistant", "content": crisis_response})
            with st.chat_message("assistant"):
                st.markdown(crisis_response, unsafe_allow_html=True)
        else:
            response = generate_response(prompt, sentiment)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# Sidebar with resources
with st.sidebar:
    st.header("Mental Health Resources")
    st.markdown("""
    - [National Suicide Prevention Lifeline](https://988lifeline.org/)
    - [Crisis Text Line](https://www.crisistextline.org/)
    - [NAMI Helpline](https://www.nami.org/help)
    """)
    st.divider()
    st.markdown("### Emotional State Analysis")
    if 'sentiment' in locals():
        st.metric(label="Detected Emotion", value=sentiment[0].capitalize())
        st.metric(label="Intensity Level", value=f"{sentiment[1]}/5")
