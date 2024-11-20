import pandas as pd
import numpy as np
import faiss
from typing import List
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from transcript_processor import ChunkedTranscript

class VectorStore:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vector_store = None
        
    def create_embeddings(self, transcripts: List[ChunkedTranscript]) -> pd.DataFrame:
        df = pd.DataFrame([{'id': t.id, 'text': t.text} for t in transcripts])
        df['embedding'] = self.embeddings.embed_documents(df['text'].tolist())
        return df
    
    def build_index(self, df: pd.DataFrame, save_path: str = None):
        embedding_matrix = np.vstack(df['embedding'].values).astype('float32')
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)
        
        if save_path:
            faiss.write_index(index, f'{save_path}.index')
            df.to_csv(f'{save_path}_metadata.csv', index=False)
        
        return index
    
    def load_index(self, path: str):
        index = faiss.read_index(f'{path}.index')
        metadata = pd.read_csv(f'{path}_metadata.csv')
        
        documents = [
            Document(page_content=text, metadata={"id": id})
            for text, id in zip(metadata['text'], metadata['id'])
        ]
        
        self.vector_store = LangchainFAISS.from_documents(
            documents, 
            self.embeddings
        )
        
        return self.vector_store 