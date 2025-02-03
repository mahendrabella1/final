import os
import requests
import tempfile
import streamlit as st
import nltk
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
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

# Function to validate URLs
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
        else:
            text_data = extract_text_from_webpage(input_path)
            if not text_data:
                return "❌ Error: No readable text found on the webpage."
            documents = [{"page_content": text_data}]
    else:
        documents = load_pdf(input_path)
    
    text_data = "\n".join([doc.page_content for doc in documents])
    
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if PINECONE_INDEX_NAME not in [index['name'] for index in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    
    PineconeVectorStore.from_texts(text_chunks, index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return "✅ Data successfully processed and stored in Pinecone."

# Query the chatbot
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

#------------------------------------------- STREAMLIT UI --------------------------------------------------

st.set_page_config(page_title="🩺 Medical-Bot", layout="wide")

# Sidebar
st.sidebar.header("📂 Data Source Selection")
data_source = st.sidebar.radio("Choose data source:", ["Upload a PDF", "Enter a URL", "Use Default Data"])

pdf_file = None
url_input = ""

if data_source == "Upload a PDF":
    pdf_file = st.sidebar.file_uploader("📂 Upload a PDF file", type=["pdf"])
elif data_source == "Enter a URL":
    url_input = st.sidebar.text_input("🔗 Paste a URL:")
elif data_source == "Use Default Data":
    st.sidebar.write("📚 Using preloaded medical data.")

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

# Chat Header
st.markdown("<h1 style='text-align: center;'>💬 Medical-Bot - AI-powered Medical Assistant</h1>", unsafe_allow_html=True)

if "queries" not in st.session_state:
    st.session_state.queries = []

st.markdown("<h3>💬 Chat History</h3>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    for q, r in st.session_state.queries:
        st.markdown(f"""
            <div style="
                border: 1px solid #444;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #262626;
            ">
                <div style="font-weight: bold; color: #0084ff;">👤 You:</div>
                <div style="margin-bottom: 8px;">{q}</div>
                <hr style="border: 0.5px solid #444;">
                <div style="font-weight: bold; color: #00d4ff;">🤖 Bot:</div>
                <div>{r}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #1e1e1e;
            padding: 15px;
            border-top: 1px solid #444;
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
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='input-container'>", unsafe_allow_html=True)
question = st.text_area("", height=80, max_chars=500, key="question_box", placeholder="Type your message...")
st.markdown("</div>", unsafe_allow_html=True)

if question:
    with st.spinner("🤔 Generating response... Please wait!"):
        try:
            response = query_chatbot(question)
            st.session_state.queries.append((question, response))
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

st.markdown("---")
st.markdown("🔍 **Medical-Bot** - AI-powered assistant for medical information. 💡")
