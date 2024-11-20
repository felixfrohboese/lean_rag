import os
from typing import List, Dict
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class ChunkedTranscript:
    id: str
    text: str

class TranscriptProcessor:
    def __init__(self, directory: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.directory = directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def validate_directory(self) -> bool:
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory {self.directory} does not exist")
        return True
    
    def process_files(self) -> List[ChunkedTranscript]:
        self.validate_directory()
        transcripts = []
        
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                transcripts.extend(self._process_single_file(filename))
        
        return transcripts
    
    def _process_single_file(self, filename: str) -> List[ChunkedTranscript]:
        chunks = []
        file_path = os.path.join(self.directory, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text_chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(text_chunks):
                    chunks.append(ChunkedTranscript(
                        id=f"{filename}_chunk_{i}",
                        text=chunk
                    ))
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            
        return chunks 