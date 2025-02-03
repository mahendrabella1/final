import os
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

# Check if the URL is valid and accessible
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
    else:
        documents = load_pdf(input_path)
        text_data = "\n".join([doc.page_content for doc in documents])
    
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if PINECONE_INDEX_NAME not in [index['name'] for index in pc.list_indexes()] :
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    
    PineconeVectorStore.from_texts(text_chunks, index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return "✅ Data successfully stored in Pinecone."

# Query chatbot with document embeddings
def query_chatbot(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    except Exception as e:
        return f"❌ Error: Could not connect to Pinecone index. {str(e)}"

    retriever = docsearch.as_retriever(search_kwargs={"k": 5})
    combine_chain = StuffDocumentsChain(prompt=PromptTemplate(input_variables=["context"], template="{context}"))
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_chain)
    response = qa_chain.run(question)
    
    return response if response else "❌ No relevant information found."
# -------------------------------------------STREAMLIT UI ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

# Set page config
st.set_page_config(page_title="🩺 Medical AI - Medical Knowledge Assistant", layout="wide")

# Apply Dark Theme
st.markdown("""
    <style>
        /* Background Color and Fonts */
        body, .stApp {
            background-color: #121212;
            color: white;
            font-family: 'Arial', sans-serif;
        }

        /* Sidebar Customization */
        .css-1d391kg {
            background-color: #1e1e1e !important;
            color: white;
        }

        /* Title Styling */
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #E0E0E0;
            text-align: center;
        }

        /* Chat Box Styling */
        .chat-box {
            background-color: #262626;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
        }

        .user-message {
            background-color: #333;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
        }

        .bot-message {
            background-color: #444;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        /* Input Box Styling */
        .stTextArea>div>textarea {
            background-color: #1e1e1e;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #6200ea;
            color: white;
            padding: 10px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
        }

        .stButton>button:hover {
            background-color: #3700b3;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("**Current Knowledge Source:**")
data_source = st.sidebar.radio("Select knowledge base:", ["Default Medical Encyclopedia", "Upload PDF", "Enter URL"])

pdf_file = None
url_input = ""

if data_source == "Upload PDF":
    pdf_file = st.sidebar.file_uploader("📂 Upload a PDF file", type=["pdf"])
elif data_source == "Enter URL":
    url_input = st.sidebar.text_input("🔗 Enter website URL:")

# Process Data Button
if st.sidebar.button("⚡ Process Data"):
    with st.spinner("📄 Processing data... Please wait!"):
        try:
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdf_file.read())
                    result = store_embeddings(tmp_file.name)
                    st.sidebar.success("✅ PDF processed successfully!")
            elif url_input and is_valid_url(url_input):
                result = store_embeddings(url_input)
                st.sidebar.success("✅ URL processed successfully!")
            elif data_source == "Default Medical Encyclopedia":
                result = store_embeddings("clinical_medicine_ashok_chandra.pdf")
                st.sidebar.success("✅ Default data processed!")
            else:
                st.sidebar.error("❌ Invalid input!")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing data: {str(e)}")

# Title & Chat Header
st.markdown("<h1 class='title'>📖 Medical AI - Medical Knowledge Assistant</h1>", unsafe_allow_html=True)
st.subheader("💬 Chat with Medical AI")

# Initialize chat history
if "queries" not in st.session_state:
    st.session_state.queries = []

# Chat Input
question = st.text_area("Ask a medical question...", height=80, max_chars=500, key="question_box")

# Submit Button
if st.button("💬 Submit Query") and question:
    with st.spinner("🤔 Generating response... Please wait!"):
        try:
            response = query_chatbot(question)
            st.session_state.queries.append((question, response))
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Chat History Section
st.subheader("📜 Chat History")

chat_history_container = st.container()
with chat_history_container:
    for q, r in st.session_state.queries:
        st.markdown(f"<div class='chat-box'><div class='user-message'><b>👤 You:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'><b>🤖 Bot:</b> {r}</div></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("🔍 **Medical AI** - AI-powered medical assistant for knowledge retrieval. 💡")

