import os
import re
import requests
import tempfile
import streamlit as st
import nltk
from dotenv import load_dotenv  # Load environment variables
from pinecone import Pinecone  # Pinecone for vector database storage
from langchain.chains import RetrievalQA  # LangChain retrieval-based Q&A system
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Pinecone vector store
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader  # PDF loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits documents into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # Embedding model
from langchain.prompts import PromptTemplate  # Custom prompt template
from bs4 import BeautifulSoup  # Web scraping for unstructured URLs
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer  # Importing for Llama2 model
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
    """Check if the provided URL is accessible."""
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Extract text from webpage
def extract_text_from_webpage(url):
    """Extract text content from a given webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()

# Load PDF and extract text
def load_pdf(pdf_path):
    """Load a PDF and extract its text."""
    return PyPDFLoader(pdf_path).load()

# Store embeddings from PDF or webpage
def store_embeddings(input_path):
    """Process text from a local PDF or an online source and store embeddings."""
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
    if PINECONE_INDEX_NAME not in [index['name'] for index in pc.list_indexes()] :
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    
    # Store the embeddings in Pinecone vector store
    PineconeVectorStore.from_texts(text_chunks, index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return "✅ Data successfully processed and stored in Pinecone."

# Query the chatbot for relevant answers using stored embeddings and language model
def query_chatbot(question):
    """Retrieves relevant information from stored embeddings and generates a response using Groq's Ollama model."""
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load Pinecone vector store
    docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

    # Initialize Groq's Ollama model
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": question,
        }],
        model="llama-3.3-70b-versatile",  # Using Groq's model
        stream=False,
    )
    
    # Get the response from Groq's Ollama model
    response = chat_completion.choices[0].message.content

    return response



#-------------------------------------------STREAMLITUI---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit interface setup
st.set_page_config(page_title="Medical-Bot", layout="wide")
st.title("🩺 Medical-Bot - AI-powered Medical Assistant")

# Sidebar for data source selection
st.sidebar.header("Data Source Selection")
data_source = st.sidebar.radio("Choose data source:", ["Upload a PDF", "Enter a URL", "Use Default Data"])

pdf_file = None
url_input = ""

if data_source == "Upload a PDF":
    pdf_file = st.sidebar.file_uploader("📂 Upload a PDF file", type=["pdf"])
elif data_source == "Enter a URL":
    url_input = st.sidebar.text_input("🔗 Paste a URL:")
elif data_source == "Use Default Data":
    st.sidebar.write("📚 Using preloaded medical data.")

# Initialize session state for storing queries and responses
if "queries" not in st.session_state:
    st.session_state.queries = []

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
            elif data_source == "Use Default Data":
                result = store_embeddings("clinical_medicine_ashok_chandra.pdf")
                st.sidebar.success("✅ Default data processed!")
            else:
                st.sidebar.error("❌ Invalid input!")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing data: {str(e)}")

# User query interface
st.header("🤖 Ask Your Medical Question")
question = st.text_area("Type your query below:", height=80, max_chars=500, key="question_box", placeholder="Ask a question...")

# Styling the input box and making it fixed at the bottom
st.markdown(
    """
    <style>
    .css-18e3th9 {
        height: 80px;
        font-size: 16px;
    }
    .stTextArea>div>textarea {
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 16px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .chat-box-container {
        max-height: 300px;
        overflow-y: auto;
        margin-bottom: 150px;
    }
    .css-1kyxreq {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    </style>
    """, unsafe_allow_html=True
)

# Submit button
if st.button("💬 Submit Query") and question:
    with st.spinner("🤔 Generating response... Please wait!"):
        try:
            response = query_chatbot(question)
            st.session_state.queries.append((question, response))
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Display Chat History
st.subheader("💬 Chat History")
chat_history = st.container()
with chat_history:
    for q, r in st.session_state.queries:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")

# Footer section
st.markdown("---")
st.markdown("🔍 **Medical-Bot** - AI-powered assistant for medical information. 💡")
