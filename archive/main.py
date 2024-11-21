import os
import dotenv
from transcript_processor import TranscriptProcessor
from vector_store import VectorStore
from query_engine import QueryEngine

def main():
    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize components
    processor = TranscriptProcessor(directory='transcripts')
    vector_store = VectorStore(api_key=api_key)
    
    # Process new transcripts if needed
    if not os.path.exists('transcripts.index'):
        print("Processing transcripts and creating new index...")
        transcripts = processor.process_files()
        df = vector_store.create_embeddings(transcripts)
        vector_store.build_index(df, save_path='transcripts')
    
    # Load existing index
    vs = vector_store.load_index('transcripts')
    
    # Initialize and run query engine
    query_engine = QueryEngine(vs, api_key=api_key)
    query_engine.run_interactive()

if __name__ == "__main__":
    main() 