import os
import streamlit as st
from chatbot import ChatBot_output
from data_loading import vectordb
from streamlit_pdf_reader import pdf_reader


def main():

    st.set_page_config(layout="wide")

    with st.sidebar:
        st.header("About")
        st.write("""
            My Name is Nitesh Sharma. I am using llama-3.1-70b-versatile model.
            ***
            github :- https://github.com/nitesh29ns/pdf_chatbot
                
            linkdin :- https://www.linkedin.com/in/nitesh-sharma-0a260b183/
            """)

    col1, col2 = st.columns(2)

    with col1:

        
        html_temp = """
        <div style="background-color:tomato;padding:5px">
        <h2 style="color:white;text-align:center;"> Talk to PDF </h2>
        </div>
        """

        st.markdown(html_temp,unsafe_allow_html=True)

        html_write_temp="""
        <div>
        <h1 style="font-size: 25px">You have to upload an pdf and ask anything with in the context of the pdf.</h1>
        </div>

        """

        st.write(html_write_temp, unsafe_allow_html=True)

        

        pdf_file = st.file_uploader("upload PDF file", type="pdf")
        
        with st.spinner("pdf to vector..."):
            if pdf_file:
                #pdf_reader(pdf_file)
                
                dir = "uploaded_pdfs"
                os.makedirs(dir,exist_ok=True)
                
                path = os.path.join(dir, pdf_file.name)
            
                with open(path, "wb") as f:
                        f.write(pdf_file.getvalue())

                chroma_db_name = f"{pdf_file.name}_db"
                #print(chroma_db_name)
                if os.path.isdir(chroma_db_name):
                    st.write("db is already exist.")
                else:
                    db = vectordb(pdf_path=path, chroma_path=chroma_db_name)
                    res = db.upload_to_vectordb()

                    st.write(res)
                st.success("Done! ✅")

        st.title("📄 PDF_Chatbot")
        query = st.chat_input()

        if "queries" not in st.session_state:
            st.session_state["queries"] = []

        if "responses" not in st.session_state:
            st.session_state["responses"] = []
        
        if query:
            st.session_state.queries.append(query)

            response = ChatBot_output(chroma_path=chroma_db_name).RAG_output(query_text=query)
            st.session_state.responses.append(response)

            for query, response in zip(st.session_state.queries, st.session_state.responses):
                with st.chat_message("user"):
                    st.write(query)
    
                with st.chat_message("assistant"):
                    st.write(response)


    with col2:
        if pdf_file:
            pdf_reader(pdf_file)


if __name__ == "__main__":
    main()

