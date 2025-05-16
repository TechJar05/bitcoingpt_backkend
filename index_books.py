# index_books.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = ""

# 1. Load all PDFs from books/
docs = []
pdf_dir = "books"
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        print(f"📘 Loading: {filename}")
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        docs.extend(loader.load())

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = splitter.split_documents(docs)

# 3. Create embeddings + FAISS index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 4. Save index
vectorstore.save_local("vector_index")
print("✅ FAISS index created from all PDFs and saved to vector_index/")
