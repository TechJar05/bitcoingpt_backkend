# index_videos.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY is missing. Please set it in your .env file.")
print("🔑 OpenAI API Key loaded successfully.")

# ✅ Full video metadata (15 prompts mapped to books and URLs)
video_metadata = [
    {
        "content": "Bitcoin as a decentralized digital currency. Peer-to-peer transactions. No banks. Limited supply. Fiat vs Bitcoin.",
        "metadata": {
            "topic": "What is Bitcoin",
            "book": "The Basics of Bitcoins and Blockchains",
            "video_url": "https://vimeo.com/1086262917/e341ef910d"
        }
    },
    {
        "content": "How Bitcoin transactions work. Inputs, outputs, digital signatures, public/private key cryptography, blockchain verification.",
        "metadata": {
            "topic": "Transactions",
            "book": "Mastering Bitcoin",
            "video_url": "https://vimeo.com/1086262945/c207154f27"
        }
    },
    {
        "content": "Bitcoin mining and Proof of Work. SHA-256 hashing, competition, nonce, difficulty, miner rewards.",
        "metadata": {
            "topic": "Mining",
            "book": "Mastering Bitcoin",
            "video_url": "https://vimeo.com/1086262967/15a8162cae"
        }
    },
    {
        "content": "Wallets and private keys. Hot vs cold storage, custodial vs self-custody, seed phrases.",
        "metadata": {
            "topic": "Wallets",
            "book": "Mastering Bitcoin",
            "video_url": "https://vimeo.com/1086263163/9146ea5567"
        }
    },
    {
        "content": "Satoshi Nakamoto and the Bitcoin whitepaper. Genesis block. Vision of peer-to-peer money.",
        "metadata": {
            "topic": "Satoshi & Whitepaper",
            "book": "The Book of Satoshi",
            "video_url": "https://vimeo.com/1086263534/145ddc3dd7"
        }
    },
    {
        "content": "Bitcoin supply limit. 21 million cap. Digital scarcity. Halvings. Inflation resistance.",
        "metadata": {
            "topic": "Limited Supply",
            "book": "The Bitcoin Standard",
            "video_url": "https://vimeo.com/1086263643/d46228eea9"
        }
    },
    {
        "content": "Sound money vs fiat. Inflation. Central banks. Bitcoin compared to gold.",
        "metadata": {
            "topic": "Sound Money",
            "book": "The Bitcoin Standard",
            "video_url": "https://vimeo.com/1086263690/51412aa99d"
        }
    },
    {
        "content": "Bitcoin nodes. Full nodes store the blockchain, validate blocks, and enforce consensus.",
        "metadata": {
            "topic": "Nodes",
            "book": "Mastering Bitcoin",
            "video_url": "https://vimeo.com/1086263751/77fab97f7d"
        }
    },
    {
        "content": "SegWit and transaction malleability. BIP141. Bitcoin scalability improvements.",
        "metadata": {
            "topic": "SegWit",
            "book": "Mastering Bitcoin",
            "video_url": "https://vimeo.com/1086263803/9693eec2ad"
        }
    },
    {
        "content": "Bitcoin in real life. Use cases in remittances, censorship resistance, economic crises.",
        "metadata": {
            "topic": "Real World Use",
            "book": "The Little Bitcoin Book",
            "video_url": "https://vimeo.com/1086263835/fc9a805ab3"
        }
    },
    {
        "content": "Early Bitcoin history. Silk Road. Mt. Gox hack. Early community and adoption.",
        "metadata": {
            "topic": "History",
            "book": "Digital Gold",
            "video_url": "https://vimeo.com/1086263859/15418db90c"
        }
    },
    {
        "content": "Bitcoin vs banking. Permissionless, borderless, no intermediaries, financial access.",
        "metadata": {
            "topic": "Banking vs Crypto",
            "book": "The Age of Cryptocurrency",
            "video_url": "https://vimeo.com/1086263887/fc6624452a"
        }
    },
    {
        "content": "How Bitcoin achieves decentralization. Open source, nodes, miner incentives, community governance.",
        "metadata": {
            "topic": "Decentralization",
            "book": "The Little Bitcoin Book",
            "video_url": "https://vimeo.com/1086263912/568e896a37"
        }
    },
    {
        "content": "Bitcoin as an investment. Volatility, long-term holding, comparing with stocks and gold.",
        "metadata": {
            "topic": "Investment Risks",
            "book": "Cryptoassets",
            "video_url": "https://vimeo.com/1086263946/408219df3e"
        }
    },
    {
        "content": "Bitcoin regulation. Government responses. ETFs, SEC, institutional adoption.",
        "metadata": {
            "topic": "Regulation",
            "book": "Cryptoassets",
            "video_url": "https://vimeo.com/1086263976/8e430d5b27"
        }
    },
]

# ✅ Convert to LangChain documents
documents = [Document(page_content=item["content"], metadata=item["metadata"]) for item in video_metadata]

# ✅ Create and save vector index
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("video_index")

print("✅ Video index built and saved to 'video_index/'")
