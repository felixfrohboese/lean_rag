# Project Overview
- Web App that allows users to upload documents and ask questions about the documents
- Combines document retrieval with GenAI based chat interface to produce accurate fact-based answers
- Makes hyperparameters configurable by end user to allow for easy tuning
- Used single but functional frontend to upload documents, tune parameters, and view chat history

# Core Functionality
- Upload txt documents to a file drop zone
  - Drag and drop one or multiple txt docuemnts
  - See uploaded files in list and have the option to delete them
  - Confirm to proceed with the uploaded files shown in the list

- Configure hyperparameters for the LLM and vector database
  - OpenAI API key from text input
  - LLM model name from dropdown menu
  - Chunk size from dropdown menu
  - Chunk overlap from dropdown menu
  - Retrieval top k from dropdown menu

- Ask questions about the documents
  - Enter a question in the text input
  - Click the submit button to see the answer
  - View chat history in the chat log

- Delete chat history button to clear the chat log, delete the uploaded files, the vector database and the hyperparameters

# Documentation



# Additional Requirements


