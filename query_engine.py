from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

class QueryEngine:
    def __init__(self, vector_store: FAISS, api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            temperature=0.1
        )
        
        prompt_template = """Use the following pieces of context to answer the question. If you cannot find the answer in the context, say so, but try to provide relevant information from the context that might be helpful.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the exact answer isn't in the context, explain what related information is available
- Quote relevant parts of the context using quotation marks
- Be concise but thorough

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.7
                }
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            }
        )
    
    def process_query(self, query: str) -> str:
        try:
            response = self.qa_chain.invoke(query)
            return response['result']
        except Exception as e:
            return f"Error processing query: {str(e)}" 