# app.py
import streamlit as st
import google.generativeai as genai
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# Theme configuration
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

# Configuration
genai.configure(api_key="AIzaSyCywgtxEx4QzNLPOk7w7czbAlMqwuNqQCo")
PSYCH_CONTENT_DIR = "psychological_resources/"

# Initialize Models
@st.cache_resource
def load_models():
    return (
        genai.GenerativeModel('gemini-2.0-flash'),
        genai.GenerativeModel('gemini-2.0-flash')
    )

chat_model, sentiment_model = load_models()

# Initialize Vector Store for RAG
@st.cache_resource(show_spinner=False)
def initialize_rag():
    try:
        loader = DirectoryLoader(
            PSYCH_CONTENT_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        return FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
    except Exception as e:
        st.error(f"RAG Initialization Error: {str(e)}")
        return None

vector_store = initialize_rag()

# Crisis detection and safety settings
CRISIS_TRIGGERS = [
    r'\b(suicide|kill myself|end it all)\b',
    r'\b(abuse|violence|rape)\b',
    r'\b(overdose|cutting|self harm)\b'
]

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm PsyAssist. How can I support you today?"}
    ]

def retrieve_psychological_context(query):
    if not vector_store:
        return ""
    results = vector_store.similarity_search(query, k=2)
    return "\n".join([f"- {doc.page_content[:300]}..." for doc in results])

def analyze_sentiment(text):
    try:
        prompt = """Analyze this message for emotional state. Respond ONLY with:
        - Primary emotion (neutral, anxious, depressed, angry, suicidal)
        - Intensity (1-5)
        - Suicidal intent (yes/no)
        Format: emotion|intensity|suicidal"""
        
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

def generate_response(text, sentiment):
    context = retrieve_psychological_context(text)
    
    prompt = f"""Generate a supportive response using these guidelines:
    {context}
    
    Requirements:
    - Respond naturally (0-50 words)
    - Focus on emotional validation
    - Suggest practical coping strategies
    - Be able to converation in friendly, non-formal and Multiple-languages based on user input.
    - Do not use any random language until user uses it.
    - Maintain conversational tone
    - Suggest various unique ways for best possible output.
    - While giving responses do not hallucinate and always check the RAG implemeneted Data or Previous chat history from user to give output.
    
    previous chat history: {st.session_state.messages}

    User message: {text}"""
    
    try:
        response = chat_model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"max_output_tokens": 200, "temperature": 2}
        )
        return response.text.strip()
    except Exception as e:
        return "I'm having trouble formulating a response right now. Please try again later."

def check_crisis(text, sentiment):
    if sentiment[2] == 'yes': 
        return True
    return any(re.search(pattern, text.lower()) for pattern in CRISIS_TRIGGERS)

# App layout
st.title("🧠 PsyAssist - Mental Health Companion")
st.caption("A safe space for emotional support and mental health resources")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
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
