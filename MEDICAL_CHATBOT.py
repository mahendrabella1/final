import os
import re
import requests
import tempfile
import streamlit as st
import nltk
from dotenv import load_dotenv  
from pinecone import Pinecone  
from langchain.chains import RetrievalQA  
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain.prompts import PromptTemplate  
from bs4 import BeautifulSoup  
import torch
from groq import Groq

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medical"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure nltk dependency is available
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to check valid URL
def is_valid_url(url):
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Extract text from webpage
def extract_text_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()

# Load PDF and extract text
def load_pdf(pdf_path):
    return PyPDFLoader(pdf_path).load()

# Store embeddings from PDF or webpage
def store_embeddings(input_path):
    if input_path.startswith("http"):
        if not is_valid_url(input_path):
            return "❌ Error: URL is not accessible."
        
        if input_path.endswith(".pdf"):
            documents = OnlinePDFLoader(input_path).load()
            text_data = "\n".join([doc.page_content for doc in documents])
        else:
            text_data = extract_text_from_webpage(input_path)
            if not text_data:
                return "❌ Error: No readable text found on the webpage."
    else:
        documents = load_pdf(input_path)
        text_data = "\n".join([doc.page_content for doc in documents])
    
    # Split text into manageable chunks
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if Pinecone index exists
    if PINECONE_INDEX_NAME not in [index['name'] for index in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    
    # Store the embeddings in Pinecone vector store
    PineconeVectorStore.from_texts(text_chunks, index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return "✅ Data successfully processed and stored in Pinecone."

# Query the chatbot for relevant answers
def query_chatbot(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    except Exception as e:
        return f"❌ Error: Could not connect to Pinecone index. {str(e)}"

    relevant_docs = docsearch.similarity_search(question, k=5)
    
    if not relevant_docs:
        return "❌ No relevant information found in stored data."

    retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant specialized in answering questions based on provided medical content."},
            {"role": "user", "content": f"Here is the relevant information from the stored data:\n\n{retrieved_text}\n\nUser's question: {question}"}
        ],
        model="deepseek-r1-distill-llama-70b",  
        stream=False,
    )

    return chat_completion.choices[0].message.content


#-------------------------------------------STREAMLIT UI---------------------------------------------------------------------------------------

# Streamlit interface setup
st.set_page_config(page_title="🩺 Medical-Bot", layout="wide")

# Apply ChatGPT-like styling with fixed input box
st.markdown("""
    <style>
        .stApp {
            background-color: #181818;
            color: white;
        }
        .css-1d391kg {
            background-color: #1e1e1e !important;
            color: white;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #E0E0E0;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-container {
            height: 75vh;
            overflow-y: auto;
            padding: 15px;
            border-radius: 10px;
            display: flex;
            flex-direction: column-reverse;
        }
        .user-message {
            background-color: #444;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
        }
        .bot-message {
            background-color: #262626;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: left;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #1e1e1e;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stTextArea>div>textarea {
            background-color: #222;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            width: 85%;
            resize: none;
            height: 80px;
        }
        .stTextArea>div::after {
            content: "➜";
            position: absolute;
            right: 15px;
            bottom: 20px;
            font-size: 24px;
            color: #0084ff;
            animation: moveArrow 1.5s infinite ease-in-out;
        }
        @keyframes moveArrow {
            0% { transform: translateX(0); }
            50% { transform: translateX(5px); }
            100% { transform: translateX(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Data Source Selection
with st.sidebar:
    st.text_input("🔑 Enter API Key", type="password", key="chatbot_api_key")
    st.header("📁 Data Source Selection")
    data_source = st.radio("Choose data source:", ["Upload a PDF", "Enter a URL", "Use Default Data"])
    pdf_file = None
    url_input = ""
    if data_source == "Upload a PDF":
        pdf_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])
    elif data_source == "Enter a URL":
        url_input = st.text_input("🔗 Paste a URL:")
    elif data_source == "Use Default Data":
        st.write("📚 Using preloaded medical data.")

# Chat Header
st.markdown("<h1 class='title'>💬 Medical-Bot</h1>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='{msg['role']}-message'>{msg['content']}</div>", unsafe_allow_html=True)

# Fixed Input Box at the Bottom
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
user_input = st.text_area("", height=80, key="input_box", placeholder="Type your message...")
st.markdown("</div>", unsafe_allow_html=True)
