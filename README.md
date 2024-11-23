# Lean RAG

AI-powered Q&A Assistant for Private Docs

## What does Lean RAG do?
Lean RAG will process your uploaded documents (txt and pdf formats), create a vector database from them and allow you to ask questions about it using the chat interface.

## How to use it:
- **Step 1 | Input Parameters**: Provide your OpenAI API along with potential changes to the pre-selected parameters for the solution. By hovering about the info sign of each input field, you get more context.
- **Step 2 | Upload Files**: Upload your documents (txt or pdf), check the list of files and click on the "Process Files" button to trigger the creation of an individual vector database for those files.
- **Step 3 | Ask Questions**: Ask questions about the content of the uploaded files in the Chat interface.
- **Important Note**: All inputs are saved in the session state, so you don't have to re-input them unless you want to change something. They will however be deleted once you close the app.

## Implementation Details

### 1. Ingestion Phase

**Process**:
- Accepts PDF and TXT files via Streamlit upload
- Extracts text content and standardizes format
- Chunks text into segments (configurable size: 512-2048 tokens)
- Adds overlap between chunks (configurable: 64-256 tokens)
**Tools**:
- PyPDF2 for PDF reading
- LangChain for text chunking


### 2. Indexing Phase

**Process**:
- Creates embeddings for each text chunk
- Stores vectors and original text in FAISS index
- Maintains in-memory vector database during session
**Tools**:
- OpenAIEmbeddings for vector creation
- FAISS for vector storage
### 3. Query Processing

**Process**:
- Converts user question to vector embedding
- Performs similarity search against stored vectors
- Retrieves top-k most relevant chunks (configurable: 3-10 chunks)
**Tools**:
- OpenAIEmbeddings for query vectorization
- FAISS similarity search

### 4. Augmentation

**Process**:
- Combines retrieved context chunks
- Constructs structured prompt with context and query
- Includes specific instructions for LLM response formatting
**Tools**:
- Custom prompt template
- Context assembly logic

### 5. Generation

**Process**:
- Sends augmented prompt to LLM
- Receives and formats response
- Displays result in Streamlit chat interface
**Tools**:
- OpenAI API (GPT-4o or GPT-4o-mini)
- Configurable temperature (0.0-1.0)