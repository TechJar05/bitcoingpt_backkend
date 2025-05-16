# # app/chat.py

# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from uuid import uuid4
# from . import models, schemas
# from .database import SessionLocal

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings  # ✅ Updated import
# from openai import OpenAI                      # ✅ New client SDK
# import os

# router = APIRouter()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Load from env

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.post("/chat/", response_model=schemas.ChatResponse)
# def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
#     # Create or fetch session
#     if req.session_id:
#         session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
#     else:
#         session = models.ChatSession(session_id=str(uuid4()))
#         db.add(session)
#         db.commit()
#         db.refresh(session)

#     # Save user message
#     user_msg = models.ChatMessage(
#         session_id=session.id,
#         role="user",
#         content=req.message.content
#     )
#     db.add(user_msg)
#     db.commit()

#     # Load vector store and get context
#     vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     docs = vectorstore.similarity_search(req.message.content, k=3)
#     context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant content found."

#     print("🔍 Context used:\n", context)

#     # Prepare prompt
#     prompt = f"""Use the context below to answer the user's question. If unrelated, say "I don't know."

# Context:
# {context}

# Q: {req.message.content}
# A:"""

#     # GPT-4-Turbo response using new SDK
#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant answering only based on the context."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     gpt_response = response.choices[0].message.content

#     # Save assistant response
#     assistant_msg = models.ChatMessage(
#         session_id=session.id,
#         role="assistant",
#         content=gpt_response
#     )
#     db.add(assistant_msg)
#     db.commit()

#     # Return chat history
#     history = db.query(models.ChatMessage)\
#         .filter_by(session_id=session.id)\
#         .order_by(models.ChatMessage.timestamp)\
#         .all()

#     formatted_history = [
#         schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
#         for m in history
#     ]

#     return schemas.ChatResponse(
#         session_id=session.session_id,
#         reply=gpt_response,
#         history=formatted_history
#     )
# app/chat.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from . import models, schemas
from .database import SessionLocal

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/chat/", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
    # ✅ Get or create session
    if req.session_id:
        session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
    else:
        session = models.ChatSession(session_id=str(uuid4()))
        db.add(session)
        db.commit()
        db.refresh(session)

    # ✅ Save user message to DB
    user_msg = models.ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message.content
    )
    db.add(user_msg)
    db.commit()

    # ✅ Auto-generate title if it's the first message
    if session.title is None:
        title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
        title_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": title_prompt}]
        )
        generated_title = title_response.choices[0].message.content.strip().strip('"')
        session.title = generated_title[:100]  # limit title length
        db.commit()

    # ✅ Load context from FAISS
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant content found."

    # ✅ JetkingGPT Prompt
    system_prompt = """
You are JetkingGPT, a certified AI tutor created to teach and explain the concepts of Bitcoin and its surrounding technologies, exclusively using 10 foundational books provided by Jetking as your knowledge source. You do not invent or speculate—you synthesize, summarize, and explain content from these books.

Your tone is clear, supportive, student-friendly, and technically sound. You are context-aware: you adjust the level of detail based on the user's knowledge (Beginner, Intermediate, or Advanced). You explain complex topics with analogies, real-world examples, and historical context wherever helpful.

Always attribute your knowledge to the 10 source books and offer citations when users ask for deeper validation. You never hallucinate or speculate beyond the scope of these books.

You do not offer:
- Financial or investment advice
- Price predictions or trading suggestions
- Wallet creation, key management, or specific platform endorsements

You avoid:
- Promoting any illegal uses of Bitcoin
- Discussing personal gains, hacks, or scam promotion
- Philosophical or political extremism

Always end your response with one of the following:
- “Would you like a simpler explanation?”
- “Shall I show you a diagram or example?”
- “Want to go deeper into this topic?”

SOURCE BOOKS:
1. Mastering Bitcoin — Andreas Antonopoulos (deep technical)
2. The Bitcoin Standard — Saifedean Ammous (economic philosophy)
3. The Book of Satoshi — Phil Champagne (primary sources)
4. The Little Bitcoin Book — Bitcoin Collective (freedom & finance)
5. Digital Gold — Nathaniel Popper (history & personalities)
6. The Age of Cryptocurrency — Vigna & Casey (macro/global lens)
7. Bitcoin Billionaires — Ben Mezrich (narrative biography)
8. How Money Got Free — Brian Eha (journalistic & political insights)
9. Cryptoassets — Burniske & Tatar (investment framework)
10. The Basics of Bitcoins and Blockchains — Antony Lewis (introductory & glossary-based)

If the user asks a question outside your bounds, gently redirect:
“I’m here to help you learn Bitcoin based on our trusted source material. I can’t help with that specific request, but I can show you how Bitcoin works in this context...”
"""

    final_prompt = f"""
Use the context below (extracted from Jetking’s 10 approved Bitcoin books) to answer the user's question.
Respond clearly and only using the information found in the context.

CONTEXT:
{context}

QUESTION:
{req.message.content}

ANSWER:
"""

    # ✅ Get polished response from GPT
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]
    )
    gpt_response = response.choices[0].message.content.strip()

    # ✅ Save assistant reply
    assistant_msg = models.ChatMessage(
        session_id=session.id,
        role="assistant",
        content=gpt_response
    )
    db.add(assistant_msg)
    db.commit()

    # ✅ Return full message history
    history = db.query(models.ChatMessage)\
        .filter_by(session_id=session.id)\
        .order_by(models.ChatMessage.timestamp)\
        .all()

    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in history
    ]

    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history
    )
