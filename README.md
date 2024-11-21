
## App Components

### transcript_processor.py
- Handles all file-related operations
- More maintainable and testable in isolation
- Could be reused for other projects needing text processing
Makes it easier to modify file handling without touching other components

### vector_store.py
- Manages all vector database operations
- Keeps embedding and indexing logic contained
- Could be swapped out for different vector store implementations
- Easier to optimize performance in isolation

### query_engine.py
- Handles the QA interface and retrieval
- Could be extended to support different interfaces (CLI, API, web)
- Makes it easier to modify the chat interface

### main.py
- Imports from these three modules
- Orchestrates the overall flow
- Handles configuration
- Provides the entry point


## App Flow

### main.py Startup
- Loads environment variables (OPENAI_API_KEY)
- Creates instances of:
    - TranscriptProcessor
    - VectorStore

### Check for Existing Index
- Looks for transcripts.index file
- If not found, starts new processing
- If found, skips to loading existing index

### If New Processing Needed:

   #### TranscriptProcessor
   transcripts = processor.process_files()
   ↓
   #### VectorStore
   df = vector_store.create_embeddings(transcripts)
   ↓
   vector_store.build_index(df, save_path='transcripts')

- Reads all .txt files from transcripts directory
    - Chunks them into smaller pieces
    - Creates embeddings using OpenAI
    - Builds and saves FAISS index

### Load Index:
- Loads the FAISS index
- Loads associated metadata
- Creates LangChain vector store
### Start Query Engine:
- Initializes QA chain with loaded index
- Starts interactive loop
For each query:
    1. Gets user input
    2. Searches vector store for relevant chunks
    3. Sends context + query to OpenAI
    4. Returns response
The data flow between components is:

Transcripts (txt files)
↓
ChunkedTranscript objects
↓
DataFrame with embeddings
↓
FAISS index
↓
Query results