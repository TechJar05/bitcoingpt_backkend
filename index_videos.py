# index_videos.py

import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ✅ Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# ✅ Optional: Debug print to confirm key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY is missing. Please set it in your .env file.")

print("🔑 OpenAI API Key loaded successfully.")

# ✅ Sample video metadata
video_metadata = [
    {
        "content": "How Bitcoin mining works using proof of work and SHA-256 hashing algorithm.",
        "metadata": {
            "book": "Mastering Bitcoin",
            "topic": "Mining",
            "video_url": "https://youtu.be/9mNgeTA13Gc?si=lEYmJiNUpr4jfraL"
        }
    },
    {
        "content": "Bitcoin wallet types including hot, cold, custodial, and self-custody with private keys.",
        "metadata": {
            "book": "Mastering Bitcoin",
            "topic": "Wallets",
            "video_url": "https://youtu.be/izvgAmbWPyE?si=Vf9DIq2OjKHKGZXb"
        }
    },
    {
        "content": "Bitcoin as a hedge against inflation in unstable economies and limited central banking access.",
        "metadata": {
            "book": "The Bitcoin Standard",
            "topic": "Inflation & currency devaluation",
            "video_url": "https://youtube.com/shorts/eVu_J7oHKgM?si=sEik4qvHuAo1Cq6H"
        }
    },
    {
        "content": "Satoshi Nakamoto's vision in the Bitcoin whitepaper and the creation of the genesis block.",
        "metadata": {
            "book": "The Book of Satoshi",
            "topic": "Satoshi's vision",
            "video_url": "https://youtube.com/shorts/WIHiQ-QwJko?si=Huj4UK_4c8Ld30UT"
        }
    }
    # ➕ Add more entries like this per video
]

# ✅ Convert to LangChain Document objects
documents = [
    Document(page_content=item["content"], metadata=item["metadata"])
    for item in video_metadata
]

# ✅ Create FAISS index and save
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("video_index")

print("✅ Video index built and saved to 'video_index/'")
