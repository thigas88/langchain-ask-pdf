from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

load_dotenv() 


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

      # loader = PyPDFLoader("mp-wallet_20241109153202_8f05.pdf")
      # document_pdf = loader.load()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Faça uma pergunta sobre seu PDF:")
      if user_question:
        documents = knowledge_base.similarity_search(user_question)
        docs_content = "\n\n".join(doc.page_content for doc in documents)

        llm = ChatOpenAI(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_BASE_URL"],
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        # llm.invoke("Why is open-source software important?")
       
        with get_openai_callback() as cb:
          input_data = {
              'input_documents': documents,
              'question': user_question,
          }
          response = chain.invoke(input=input_data)
          print(cb)
           
        st.write(response)

        # for chunk in chain.stream(response):
        #   print(chunk)
    

if __name__ == '__main__':
    main()
