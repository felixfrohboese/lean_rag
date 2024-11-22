from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import logging

class QueryEngine:
    def __init__(self, vector_store: FAISS, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            temperature=0.1
        )
        
        prompt_template = """Use the following pieces of context to answer the question. If you cannot find the answer in the context, say so, but try to provide relevant information from the context that might be helpful.

Context: {context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the exact answer isn't in the context, explain what related information is available
- Quote relevant parts of the context using quotation marks
- Be concise but thorough

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "query"]
        )
        
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Create the chain using the new interface with proper formatting
        self.qa_chain = (
            {
                "context": lambda x: "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(x["query"])),
                "query": lambda x: x["query"]
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
    
    def preprocess_query(self, query: str) -> str:
        system_prompt = """You are a helpful assistant that improves search queries. 
        Your task is to rewrite the given query to be more specific and detailed for better document retrieval.
        Maintain the core intent but add relevant context and specificity.
        Keep the rewritten query concise and focused."""
        
        user_prompt = f"""Original query: "{query}"
        Rewrite this query to be more effective for semantic search while maintaining its original intent."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def process_query(self, query_text):
        try:
            # Preprocess the query for better results
            enhanced_query = self.preprocess_query(query_text)
            
            # Invoke the chain with the enhanced query
            response = self.qa_chain.invoke({"query": enhanced_query})
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise Exception(f"Error processing query: {str(e)}") 