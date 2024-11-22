import streamlit as st
import os
import tempfile
from transcript_processor import TranscriptProcessor
from vector_store import VectorStore
from query_engine import QueryEngine

st.title("Easy RAG")

# Add user instructions below the title
st.markdown("""
## AI-powered Q&A Assistant for Private Docs
            
### What does Easy RAG do?
Easy RAG will process your uploaded text files (txt format), create a vector database from them and allow you to ask questions about it using the chat interface.
            
### How to use it:
- **Step 1 | Input Parameters**: Provide your OpenAI API along with potential changes to the pre-selected parameters for the solution. By hovering about the info sign of each input field, you get more context. 
- **Step 2 | Upload Files**: Upload your text files, check the list of files and click on the "Process Files" button to trigger the creation of an individual vector database for those files.
- **Step 3 | Ask Questions**: Ask questions about the content of the uploaded files in the Chat interface.
- **Important Note**: All inputs are saved in the session state, so you don't have to re-input them unless you want to change something. They will however be deleted once you close the app.
            

---
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Request OpenAI API Key
api_key = st.text_input("OpenAI API Key", type="password", help="OpenAI API key to bill your usage of the LLM. Remember to enable limits. Get your API key from https://platform.openai.com/settings/profile/api-keys")


# Model configuration
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox(
        "Large Language Model (LLM)", 
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="""
        The model to use for the LLM. 
        gpt-4o is the most recent general model. 
        gpt-4o-mini is its more lightweight and cost effective alternative."""
    )
with col2:
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="Temperature controls randomness in responses. Lower values are more focused and deterministic. 0.1 is a good value for most cases."
    )

# Retrieval and chunking configuration
col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.selectbox(
        "Number of Retrieved Documents",
        [3, 5, 7, 10],
        index=1,
        help="Number of most relevant document chunks to retrieve for augmenting the LLM response by the retrieved context. 3 is a good value for most cases."
    )
with col2:
    chunk_size = st.selectbox(
        "Chunk Size", 
        [512, 1024, 2048], 
        index=1,
        help="Size of text chunks for processing. The size of the chunks is a trade-off between the retrieval quality and the processing time. 1024 is a good value for most cases."
    )
with col3:
    chunk_overlap = st.selectbox(
        "Chunk Overlap", 
        [64, 128, 256], 
        index=1,
        help="Overlap between consecutive chunks. The overlap is a trade-off between the retrieval quality and the processing time. 128 is a good value for most cases."
    )

# File uploader section
uploaded_files = st.file_uploader(
    "Upload Text Files (.txt only)", 
    type=['txt'], 
    accept_multiple_files=True,
    help="""
    Upload your text files to be processed. The files should be in txt format.
    """
)

if uploaded_files:
    if st.button("Process Files"):
        with st.spinner("Processing files and creating vector database..."):
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temporary directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                
                # Pass these parameters when initializing the processor
                processor = TranscriptProcessor(
                    directory=temp_dir,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                transcripts = processor.process_files()
                
                # Create vector store
                vector_store = VectorStore(api_key=api_key)
                df = vector_store.create_embeddings(transcripts)
                st.session_state.vector_store = vector_store.build_vector_store(df)
                
                # Initialize query engine with new parameters
                st.session_state.query_engine = QueryEngine(
                    st.session_state.vector_store, 
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    top_k=top_k
                )
                
            st.success("Files processed successfully!")

# Chat interface
if st.session_state.query_engine is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.query_engine.process_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 