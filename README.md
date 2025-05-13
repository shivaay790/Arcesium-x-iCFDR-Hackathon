# 🧠 PsyAssist

**PsyAssist** is an intelligent, multimodal mental health chatbot that leverages **Retrieval-Augmented Generation (RAG)** and **tonality-aware interaction** to offer personalized, empathetic support. Built with Flask, it features a smooth and user-friendly interface, making conversations feel natural and human-centered.

![Screenshot (2422)](https://github.com/user-attachments/assets/157f68a4-62d1-41b6-87a7-822da9ca9fd9)

## ✨ Features

- 🔍 **RAG-Based Retrieval**  
  Pulls contextually relevant mental health content from a custom PDF knowledge base using vector embeddings.

- 🎭 **Tonality Awareness**  
  Dynamically adjusts responses based on the emotional tone detected in user input.

- 🧠 **Multimodal Capabilities**  
  Designed to support multiple types of input and expression for a richer conversation.

- 🖥️ **Modern, Clean UI**  
  Built with HTML/CSS/JS for a seamless and calming user experience.

## 📁 Project Structure
```
├── index.html # Main chatbot UI
├── help_fnq.html # Help / FAQ page
├── style.css # Frontend styling
├── script.js # Client-side logic
├── app.py # Flask backend for routing and logic
├── rag_embeddings.py # Embeds PDFs into vector database
├── resources/ # Miscellaneous assets and helper files
└── rag_database/ # (You create this) Folder to store mental health PDFs
```

![image](https://github.com/user-attachments/assets/9daf659a-4629-4bfe-81d4-53984ccd9d3e)



## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/PsyAssist.git
cd PsyAssist
mkdir rag_database
pip install -r requirements.txt
```

### 2. Add api keys

### 3. Add pdfs related to mental health into rag_database folder to help the agent.

### 4. Run the Python File
- python app.py
- Default Flask Address: https://localhost:5000

Pull requests are welcome. Fork the repo and submit your ideas or improvements.

