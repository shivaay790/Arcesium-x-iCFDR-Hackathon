import os
import re
import json
import time
import uuid
from datetime import datetime
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings  

import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='.')

# Configuration
api_key = os.getenv("API_KEY")
genai.configure(api_key= api_key)  #Add your API Key
PSYCH_CONTENT_DIR = "rag_database/"
FAISS_DB_DIR = "rag_database/faiss_index"
CHAT_HISTORY_DIR = "chat_history/"

try:
    chat_model = genai.GenerativeModel('gemini-2.0-flash')
    sentiment_model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Safety settings for the API
    SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    # Crisis trigger patterns
    CRISIS_TRIGGERS = [
        r'suicid(e|al)',
        r'kill (myself|me)',
        r'want to die',
        r'end (my|this) life',
        r'hurt (myself|me)',
        r'harm (myself|me)',
        r'don\'t want to (live|be alive|exist)'
    ]
    
    print("AI models initialized successfully")
except Exception as e:
    print(f"Error initializing AI models: {str(e)}")
    print("The application will have limited functionality")


# Load the vector database
def load_vector_db():
    """Load the vector database of psychological content using FAISS"""
    db_path = FAISS_DB_DIR
    
    if not os.path.exists(db_path):
        print("ERROR: Vector database not found.")
        print("Please run rag_embeddings.py first to create the database.")
        return None
    
    try:
        print("Loading FAISS vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(db_path, embeddings)  # Changed to FAISS.load_local
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
        return None

# Initialize the vector database
vector_db = load_vector_db()

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)
   

def retrieve_psychological_context(query):
    """Retrieve relevant psychological context from the vector database"""
    if not vector_db:
        print("No vector database available.")
        return "No specific psychological information available."
    
    try:
        results = vector_db.similarity_search(query, k=2)  # API is the same for FAISS
        context = "\n".join([f"- {doc.page_content[:300]}..." for doc in results])
        return context
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return "Unable to retrieve psychological context at this time."
def analyze_sentiment(text):
    """Analyze the sentiment of text to detect emotion and intensity"""
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
        if len(parts) == 3:
            return parts
        else:
            print("Unexpected sentiment analysis format. Using defaults.")
            return ["neutral", "1", "no"]
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return ["neutral", "1", "no"]

def check_crisis(text, sentiment):
    """Check if the text or sentiment indicates a crisis situation"""
    # Check if sentiment analysis detected suicidal intent
    if sentiment[2] == 'yes': 
        return True
    
    # Check for crisis keywords
    return any(re.search(pattern, text.lower()) for pattern in CRISIS_TRIGGERS)

# Chat history stored by session ID (in-memory) this helps chatbot to keep contexts
chat_sessions = {}

# Persistent chat history management
def save_chat_session(session_id, messages):
    """Save a chat session to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CHAT_HISTORY_DIR}/{timestamp}_{session_id[:8]}.json"
    
    metadata = {
        "session_id": session_id,
        "timestamp": timestamp,
        "message_count": len(messages),
        "summary": generate_session_summary(messages),
        "messages": messages
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up old sessions if there are more than 10
    cleanup_old_sessions()
    
    return filename

def cleanup_old_sessions():
    """Keep only the 10 most recent chat sessions"""
    files = [os.path.join(CHAT_HISTORY_DIR, f) for f in os.listdir(CHAT_HISTORY_DIR) 
             if f.endswith('.json')]
    
    if len(files) > 10:
        files.sort(key=os.path.getmtime)
        for old_file in files[:-10]:
            os.remove(old_file)

def generate_session_summary(messages):
    """Generate a short summary of the chat session"""
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return "Empty chat session"
        
    first_message = user_messages[0]
    return first_message[:50] + "..." if len(first_message) > 50 else first_message

def load_recent_sessions(limit=2):
    """Load the most recent chat sessions"""
    files = [os.path.join(CHAT_HISTORY_DIR, f) for f in os.listdir(CHAT_HISTORY_DIR) 
             if f.endswith('.json')]
    
    if not files:
        return []
    
    files.sort(key=os.path.getmtime, reverse=True)
    recent_files = files[:limit]
    
    sessions = []
    for file_path in recent_files:
        try:
            with open(file_path, 'r') as f:
                session = json.load(f)
                sessions.append(session)
        except Exception as e:
            print(f"Error loading session file {file_path}: {e}")
    
    return sessions

def get_chat_context(session_id, current_message):
    """Get context from current and past chat sessions for the model"""
    session_history = chat_sessions.get(session_id, [])
    current_context = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in session_history[-5:]  # Last 5 messages from current session
    ])
    
    # Get past session context
    past_context = ""
    past_sessions = load_recent_sessions(2)
    
    if past_sessions:
        past_context = "Previous conversations:\n"
        
        for idx, session in enumerate(past_sessions):
            past_context += f"Conversation {idx+1}:\n"
            
            # Get last 3 message pairs from past sessions
            messages = session.get("messages", [])
            if messages:
                # Group into QA pairs
                pairs = []
                for i in range(0, len(messages)-1, 2):
                    if i+1 < len(messages):
                        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                            pairs.append((messages[i]["content"], messages[i+1]["content"]))
                
                # Get last 3 pairs
                for user_msg, asst_msg in pairs[-3:]:
                    past_context += f"User: {user_msg[:100]}...\n"
                    past_context += f"Assistant: {asst_msg[:100]}...\n"
            
            past_context += "\n"
    
    return current_context, past_context

def generate_response(text, sentiment, session_id, tone="friendly"):
    """Generate a supportive response using psychological context and chat history"""
    context = retrieve_psychological_context(text)
    
    # Get current and past chat context
    current_context, past_context = get_chat_context(session_id, text)
    
    # Add tone-specific guidance based on user selection
    tone_guidance = ""
    if tone == "friendly":
        tone_guidance = "warm, conversational, and approachable"
    elif tone == "professional":
        tone_guidance = "clear, structured, and factual with professional advice"
    elif tone == "empathetic":
        tone_guidance = "deeply understanding, validating feelings with compassionate language"
    elif tone == "motivational":
        tone_guidance = "encouraging, positive, focusing on strengths and inspiring action"
    else:
        tone_guidance = "warm and conversational"
    
    prompt = f"""Generate a supportive response using these guidelines:
    
    Relevant psychological information:
    {context}
    
    Current conversation:
    {current_context}
    
    {past_context}
    
    Requirements:
    - Respond naturally (3-4 sentences)
    - Focus on emotional validation for their {sentiment[0]} emotions
    - Suggest practical coping strategies
    - Maintain a {tone} tone: {tone_guidance}
    - If the user mentions something they've discussed in previous conversations, acknowledge it
    - Do not use any language other than English unless the user initiates in another language
    - Be specific and helpful, not generic
    
    User message: {text}"""
    
    try:
        print(f"Generating response with tone: {tone}")
        response = chat_model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config={"max_output_tokens": 250, "temperature": 0.7}
        )
        return response.text.strip()
    except Exception as e:
        print(f"Response generation error: {str(e)}")
        return "I'm having trouble formulating a response right now. Please try again later."
# API endpoints
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        end_chat = data.get('end_chat', False)  
        tone = data.get('tone', 'friendly')  
        
        # Initialize session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Add user message to history
        chat_sessions[session_id].append({"role": "user", "content": user_input})
        
        # Process message
        print(f"Processing message: {user_input}")
        sentiment = analyze_sentiment(user_input)
        print(f"Detected emotion: {sentiment[0]} (Intensity: {sentiment[1]}/5)")
        
        if check_crisis(user_input, sentiment):
            crisis_response = """I notice you're expressing some concerning thoughts. 
            Please remember that you're not alone, and help is available.
            
            If you're in crisis, consider reaching out to:
            - National Suicide Prevention Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            - Your local emergency services
            
            Would you like me to suggest some coping strategies that might help right now?"""
            
            chat_sessions[session_id].append({"role": "assistant", "content": crisis_response})
            
            return jsonify({
                'response': crisis_response,
                'crisis': True,
                'session_id': session_id
            })
        else:
            response = generate_response(user_input, sentiment, session_id, tone)
            

            chat_sessions[session_id].append({"role": "assistant", "content": response})
            
            # Prune history if too long
            if len(chat_sessions[session_id]) > 20:
                chat_sessions[session_id] = chat_sessions[session_id][-20:]
                
            # If end_chat is True, save the session to disk
            if end_chat and len(chat_sessions[session_id]) > 2:  # Only save if there was an actual conversation
                save_file = save_chat_session(session_id, chat_sessions[session_id])
                print(f"Chat session saved to {save_file}")
                
            return jsonify({
                'response': response,
                'crisis': False,
                'session_id': session_id
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'response': "I'm having trouble connecting. Please try again later.",
            'error': str(e)
        }), 500

@app.route('/new-chat', methods=['POST'])
def new_chat():
    data = request.json
    old_session_id = data.get('session_id')
    
    # Save previous session if it exists and has messages
    if old_session_id and old_session_id in chat_sessions and len(chat_sessions[old_session_id]) > 2:
        save_file = save_chat_session(old_session_id, chat_sessions[old_session_id])
        print(f"Previous chat session saved to {save_file}")
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    chat_sessions[new_session_id] = []
    
    return jsonify({
        'session_id': new_session_id
    })

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    sessions = load_recent_sessions(5) 
    
    # Formating the sessions for display
    history = []
    for session in sessions:
        history.append({
            'id': session.get('session_id', 'unknown'),
            'timestamp': session.get('timestamp', 'unknown'),
            'summary': session.get('summary', 'No summary available'),
            'message_count': session.get('message_count', 0)
        })
    
    return jsonify({
        'history': history
    })



@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files like CSS, JS and images"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT environment variable
    print(f"Starting PsyAssist on http://0.0.0.0:{port}/")
    
    if not vector_db:
        print("\nWARNING: Vector database not loaded!")
        print("RAG functionality will be limited.")
        print("Please run rag_embeddings.py to create the database first.\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)

