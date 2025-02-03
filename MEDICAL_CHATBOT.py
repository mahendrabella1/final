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

