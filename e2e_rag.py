#!/Users/felixfrohboese/opt/anaconda3/bin/python3.11

import sys
print(sys.executable)
print(sys.path)

import os
import pandas as pd
import dotenv
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
import numpy as np

from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document  # Add this import
from langchain_openai import OpenAIEmbeddings  # Change this import
from langchain_openai import ChatOpenAI  # Add this import

dotenv.load_dotenv()    
openai_api_key = os.getenv('OPENAI_API_KEY')

#DATA PREPARATION

transcripts = []
directory = 'transcripts'

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            # Basic cleaning (customize as needed)
            text = text.replace('\n', ' ').strip()
            transcripts.append({'id': filename, 'text': text})

df = pd.DataFrame(transcripts)



#EMBEDDINGS

embeddings = OpenAIEmbeddings()
df['embedding'] = embeddings.embed_documents(df['text'].tolist())



#VECTOR STORAGE

# Convert embeddings to a numpy array
embedding_matrix = np.vstack(df['embedding'].values).astype('float32')

# Initialize FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save index and metadata
faiss.write_index(index, 'transcripts.index')
df.to_csv('transcripts_metadata.csv', index=False)


#LOADING THE INDEX AND METADATA

# Load FAISS index and metadata
index = faiss.read_index('transcripts.index')
metadata = pd.read_csv('transcripts_metadata.csv')

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a list of Document objects
documents = [
    Document(page_content=text, metadata={"id": id})
    for text, id in zip(metadata['text'], metadata['id'])
]

# Initialize LangChain's FAISS wrapper
vector_store = LangchainFAISS.from_documents(documents, embeddings)

# Set up RetrievalQA chain
llm = ChatOpenAI(model_name="gpt-4o-mini")  # Use ChatOpenAI instead of OpenAI
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)


#COMMAND LINE INTERFACE
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = qa_chain.invoke(query)  # Use invoke instead of run
    print("Response:", response['result'])  # Access the 'result' key of the response
