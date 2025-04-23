import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("api_key"))

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("nutrition_faiss_index")

# Create conversational chain for answering nutrition-related questions
def get_conversational_chain():
    prompt_template = """
    You are a certified AI Nutritionist. Use the provided context to answer the question with accurate,
    science-backed information. If the context does not contain the answer, say "Answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro")  # Or another model if preferred
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="AI Nutritionist", layout="centered")
    st.title("🧠 AI Nutritionist")
    st.write("Upload your nutrition-related PDFs and ask questions!")

    pdf_docs = st.file_uploader("Upload Nutrition PDFs", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Documents processed and vector store created!")

    query = st.text_input("Ask a nutrition question:")
    if query:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import GoogleGenerativeAIEmbeddings
        
        vectorstore = FAISS.load_local("nutrition_faiss_index", embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        docs = vectorstore.similarity_search(query)
        chain = get_conversational_chain()
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

if __name__ == "__main__":
    main()
