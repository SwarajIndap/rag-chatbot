import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import tempfile, os, time
from dotenv import load_dotenv

load_dotenv()

st.title("RAG Chatbot - Ask Your PDF")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # Limit chunks to avoid free-tier rate limits on large PDFs
    chunks = chunks[:40]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    with st.spinner("Indexing PDF... this may take a minute"):
        vectorstore = None
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            time.sleep(2)  # avoid hitting rate limit

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.success(answer)

    os.unlink(tmp_path)
