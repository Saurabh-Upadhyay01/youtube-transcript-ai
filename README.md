# 🎬 YouTube Transcript AI

### Watch Less. Understand More. Ask Anything.

**YouTube Transcript AI** is a RAG-based AI chatbot that extracts YouTube transcripts and answers questions using LangChain, HuggingFace, FAISS, and Groq.

---

## 🚀 Features

* 🎥 **YouTube Transcript Extraction** – Extract transcripts directly from YouTube videos.
* 💬 **AI-Powered Chatbot** – Ask questions about video content.
* 🧠 **RAG Architecture** – Generates answers based on retrieved transcript information.
* 🔍 **Semantic Search** – Finds relevant content using vector similarity.
* ⚡ **Fast AI Responses** – Uses Groq for LLM inference.
* 📚 **Text Processing** – Splits transcripts into smaller chunks for efficient retrieval.
* 🔐 **Secure API Configuration** – Stores API keys using environment secrets.

## 🛠️ Tech Stack

| Technology             | Purpose               |
| ---------------------- | --------------------- |
| Python                 | Backend development   |
| Streamlit              | User interface        |
| YouTube Transcript API | Transcript extraction |
| LangChain              | RAG pipeline          |
| HuggingFace            | Text embeddings       |
| all-MiniLM-L6-v2       | Embedding model       |
| FAISS                  | Vector database       |
| Groq                   | LLM inference         |
| GitHub                 | Version control       |
| Render                 | Cloud deployment      |

## 🏗️ How It Works

```text
YouTube Video URL
       |
       ▼
Transcript Extraction
       |
       ▼
Text Chunking
       |
       ▼
HuggingFace Embeddings
       |
       ▼
FAISS Vector Database
       |
       ▼
User Question
       |
       ▼
Similarity Search
       |
       ▼
Relevant Transcript Chunks
       |
       ▼
Groq LLM
       |
       ▼
Context-Aware Answer
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Saurabh-Upadhyay01/youtube-transcript-ai.git
```

### 2. Navigate to the Project

```bash
cd youtube-transcript-ai
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Groq API Key

Create a file named:

`.streamlit/secrets.toml`

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Get your API key from [Groq Console](https://console.groq.com/).

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🔑 Environment Variables

| Variable       | Description                    |
| -------------- | ------------------------------ |
| `GROQ_API_KEY` | API key for Groq LLM inference |

Never upload your API keys or secrets to GitHub.

## 📌 Limitations

* Transcript extraction requires an available YouTube transcript.
* YouTube may block requests originating from cloud-hosted IP addresses.
* FAISS storage is in-memory and must be rebuilt after a session or application restart.
* AI responses depend on the quality and availability of the retrieved transcript.

## 🔮 Future Enhancements

* Multilingual transcript support.
* PDF document question answering.
* Multiple YouTube video support.
* Timestamp-based answers.
* Video summarization.
* Modern frontend using HTML, CSS and JavaScript.

## 👨‍💻 Developer

**Saurabh Upadhyay**

BCA – Data Science

GitHub: [@Saurabh-Upadhyay01](https://github.com/Saurabh-Upadhyay01)

## 📄 License

This project is licensed under the MIT License.
