# What is this app?
This simple RAG app will process the uploaded text files, create a vector database and allow you to ask questions about it.

# How to use it:
- **Step 1**: Provide your OpenAI API key to use the selected gpt-4o-mini model, which is the most recent and cost effective LLM by OpenAI
- **Step 2**: Upload your text files, check the list of files and click on the "Process Files" button to trigger the creation of a database
- **Step 3**: Ask questions about the content of the uploaded files in the Chat-like interface.

## How it technically works:
- The app uses a combination of semantic search and a retrieval-augmented generation (RAG) pipeline to answer your questions.
- The semantic search helps find the most relevant chunks of text from the uploaded files, while the RAG pipeline generates a detailed answer using the context from the semantic search.
- The RAG pipeline is powered by a large language model (LLM) that is designed to understand and generate human-like text.

## What it can do: 
- **Multiple languages**: The LLM is also capable of understanding and generating text in multiple languages, so you can ask questions in different languages.
- **Complex questions**: The RAG pipeline is also capable of handling complex questions and providing detailed answers.
- **Comprehensive answers**: The semantic search and RAG pipeline work together to provide you with a comprehensive answer to your question.
- Relevant context: The semantic search helps narrow down the most relevant chunks of text from the uploaded files, while the RAG pipeline generates a detailed answer using the context from the semantic search.
