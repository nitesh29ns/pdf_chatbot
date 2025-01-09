from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from data_loading import get_embedding_funcation
import streamlit as st

# base prompt
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---


Answer the question based on the above context: {question}
"""

# return chatbot response
class ChatBot_output():
    def __init__(self,chroma_path:str):
        try:
            self.chroma_path = chroma_path
        except Exception as e:
            raise e
        
    def RAG_output(self,query_text:str):
        embedding_function = get_embedding_funcation() # embedding
        db =Chroma(persist_directory= self.chroma_path, #chroma DB
                embedding_function=embedding_function)
        # top 5 similar results based on the query
        Results = db.similarity_search_with_score(query_text, k=5) 

        context_text = "\n\n---\n\n".join([doc.page_content for doc ,_score in Results])

        # final prompt for llm model
        prompt_tamplate = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        prompt = prompt_tamplate.format(context=context_text, question=query_text)
        #print(prompt)

        # using  llama-3.1 70b model through graq api
        llm = ChatGroq(
                temperature=0,
                groq_api_key=  st.secrets.GROQ_API_KEY,
                model_name="llama-3.1-70b-versatile"
            )

        # final response by llm model
        response_text = llm.invoke(prompt)

        
        return response_text.content
    
"""
chatbot = ChatBot_output(chroma_path="./test")

response = chatbot.RAG_output(query_text="minimum age of playing?")

print(response)
"""

