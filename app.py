import streamlit as st
import os
import tempfile
from transcript_processor import TranscriptProcessor
from vector_store import VectorStore
from query_engine import QueryEngine

st.title("Document Q&A Assistant")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Request OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    # Model configuration
    model_name = st.selectbox(
        "Select Model", 
        ["gpt-4o", "gpt-4o-mini"],
        index=0
    )
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="Controls randomness in responses. Lower values are more focused and deterministic."
    )
    
    # Retrieval configuration
    top_k = st.selectbox(
        "Number of Retrieved Documents",
        [3, 5, 7, 10],
        index=1,
        help="Number of most relevant document chunks to retrieve"
    )
    
    # Chunking configuration
    chunk_size = st.selectbox(
        "Chunk Size", 
        [512, 1024, 2048], 
        index=1,
        help="Size of text chunks for processing"
    )
    chunk_overlap = st.selectbox(
        "Chunk Overlap", 
        [64, 128, 256], 
        index=1,
        help="Overlap between consecutive chunks"
    )

    # File uploader section
    uploaded_files = st.file_uploader(
        "Upload text files", 
        type=['txt'], 
        accept_multiple_files=True
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