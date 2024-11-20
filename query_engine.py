from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

class QueryEngine:
    def __init__(self, vector_store: FAISS, api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
        )
    
    def process_query(self, query: str) -> str:
        try:
            response = self.qa_chain.invoke(query)
            return response['result']
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def run_interactive(self):
        print("Starting interactive query session (type 'exit' to quit)")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            response = self.process_query(query)
            print("\nResponse:", response) 