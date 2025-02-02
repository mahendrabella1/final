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
from langchain.chains.combine_documents import StuffDocumentsChain
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

# Streamlit UI
st.set_page_config(page_title="Medical-Bot", layout="wide")
st.title("🩺 Medical-Bot - AI-powered Medical Assistant")

data_source = st.sidebar.radio("Choose data source:", ["Upload a PDF", "Enter a URL", "Use Default Data"])

pdf_file = None
url_input = ""

if data_source == "Upload a PDF":
    pdf_file = st.sidebar.file_uploader("📂 Upload a PDF file", type=["pdf"])
elif data_source == "Enter a URL":
    url_input = st.sidebar.text_input("🔗 Paste a URL:")
elif data_source == "Use Default Data":
    st.sidebar.write("📚 Using preloaded medical data.")

if "queries" not in st.session_state:
    st.session_state.queries = []

if st.sidebar.button("⚡ Process Data"):
    with st.spinner("📄 Processing data..."):
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
        except Exception as e:
            st.sidebar.error(f"❌ Error processing data: {str(e)}")

st.header("🤖 Ask Your Medical Question")
question = st.text_area("Type your query below:", height=80, max_chars=500, key="question_box", placeholder="Ask a question...")

if st.button("💬 Submit Query") and question:
    with st.spinner("🤔 Generating response..."):
        try:
            response = query_chatbot(question)
            st.session_state.queries.append((question, response))
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

st.subheader("💬 Chat History")
for q, r in st.session_state.queries:
    st.write(f"**You:** {q}")
    st.write(f"**Bot:** {r}")

st.markdown("---")
st.markdown("🔍 **Medical-Bot** - AI-powered assistant for medical information. 💡")
