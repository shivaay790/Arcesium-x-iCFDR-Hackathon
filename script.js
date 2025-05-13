// Global variables
let sessionId = null;
let currentTone = "friendly";
let shouldSaveSession = true;

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-message');
    const sendButton = document.getElementById('send-btn');
    const crisisResources = document.getElementById('crisis-resources');
    
    // Mobile sidebar toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    menuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 768 && 
            !sidebar.contains(e.target) && 
            !menuToggle.contains(e.target) &&
            sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
        }
    });
    
    // Function to add a message to the chat
    function addMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
        
        const messagePara = document.createElement('p');
        messagePara.textContent = message;
        messageDiv.appendChild(messagePara);
        
        if (!isUser) {
            const disclaimer = document.createElement('small');
            disclaimer.textContent = 'Remember: I\'m an AI assistant, not a replacement for professional mental health care.';
            messageDiv.appendChild(disclaimer);
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send message to server
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input
        userInput.value = '';
        
        try {
            // Send message to server with session ID and tone
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: sessionId,
                    tone: currentTone // Include the tone parameter
                }),
            });
            
            const data = await response.json();
            
            // Update session ID if provided
            if (data.session_id) {
                sessionId = data.session_id;
            }
            
            // Display bot response
            addMessage(data.response, false);
            
            // Show crisis resources if crisis detected
            if (data.crisis) {
                crisisResources.style.display = 'block';
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage("I'm having trouble connecting. Please try again later.", false);
        }
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // NEW CHAT button functionality
    document.querySelector('.btn').addEventListener('click', function() {
        // Clear chat messages except the first welcome message
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        // Hide crisis resources
        if (crisisResources) {
            crisisResources.style.display = 'none';
        }
        
        // Send current session ID to be saved before creating a new one
        startNewSession(sessionId);
    });
    
    // Add tone setting functionality
    const toneLinks = document.querySelectorAll('.dropdown-content a');
    toneLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            // Check if this is a tone selection link (parent's button contains "Tone")
            const parentButton = link.parentElement.previousElementSibling;
            if (!parentButton || !parentButton.textContent.includes('Tone')) {
                return; // Skip if not a tone link
            }
            
            const tone = this.textContent.toLowerCase();
            currentTone = tone; // Set the current tone
            
            // Update button text to show selected tone
            parentButton.innerHTML = `<i class="fas fa-sliders-h"></i> ${this.textContent}`;
            
            // Provide user feedback
            addMessage(`I'll adjust my tone to be more ${tone} now.`, false);
            console.log("Tone set to:", currentTone);
        });
    });
    
    // Load chat history
    loadChatHistory();
    
    // When user navigates away, save the session
    window.addEventListener('beforeunload', function() {
        if (sessionId && shouldSaveSession) {
            navigator.sendBeacon('/chat', JSON.stringify({
                message: "_session_end_",
                session_id: sessionId,
                tone: currentTone, // Include the tone parameter
                end_chat: true
            }));
        }
    });
    
    // Start a session when the page loads
    startNewSession();
});

// Function to start a new chat session
async function startNewSession(oldSessionId = null) {
    try {
        const response = await fetch('/new-chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                session_id: oldSessionId
            }),
        });
        const data = await response.json();
        sessionId = data.session_id;
        console.log("New session started:", sessionId);
    } catch (error) {
        console.error("Error starting new session:", error);
    }
}

// Add a function to load chat history
async function loadChatHistory() {
    try {
        const response = await fetch('/chat-history');
        const data = await response.json();
        
        // Clear existing history items
        const historyDropdown = document.querySelector('.dropdown-content');
        
        if (!historyDropdown) return;
        
        // Remove all but the "View All History" option
        while (historyDropdown.children.length > 1) {
            historyDropdown.removeChild(historyDropdown.firstChild);
        }
        
        // Add history items
        data.history.forEach(session => {
            const date = new Date(session.timestamp.replace('_', 'T').replace(/_/g, ':')).toLocaleString();
            const historyItem = document.createElement('a');
            historyItem.href = '#';
            historyItem.textContent = `${session.summary} (${date})`;
            
            // Insert at the beginning
            if (historyDropdown.firstChild) {
                historyDropdown.insertBefore(historyItem, historyDropdown.firstChild);
            } else {
                historyDropdown.appendChild(historyItem);
            }
        });
    } catch (error) {
        console.error("Error loading chat history:", error);
    }
}