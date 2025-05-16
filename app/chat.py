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

    # ✅ Save user message
    user_msg = models.ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message.content
    )
    db.add(user_msg)
    db.commit()

    # ✅ Auto-title
    if session.title is None:
        title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
        title_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": title_prompt}]
        )
        session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
        db.commit()

    # ✅ Load FAISS context
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # ✅ Prompt behavior
    system_prompt = """
You are JetkingGPT, a certified AI tutor created to teach and explain the concepts of Bitcoin and its surrounding technologies. You are trained only on knowledge derived from 10 foundational Bitcoin books provided by Jetking.

You do not speculate, hallucinate, or expose book titles or authors. If a user question falls outside the Bitcoin context, reframe it into a relevant Bitcoin-related question and ask that reframed question back to the user.

Always speak in a helpful, student-friendly tone. Avoid mentioning sources. Keep answers clear, neutral, and contextual.

Examples:

❌ User: Should I buy stocks?
✅ Reframe: Would you like to explore how Bitcoin compares to traditional investments?

❌ User: Can I buy drugs with crypto?
✅ Reframe: Let’s focus on how Bitcoin’s public ledger helps ensure transparency and traceability. Would you like to learn how transactions are recorded?

At the end of a valid answer, include one of:
- “Would you like a simpler explanation?”
- “Shall I show you a diagram or example?”
- “Want to go deeper into this topic?”
"""

    if context.strip() == "":
        # 🧠 No matching book content — reframe the question
        reframed = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The user asked: \"{req.message.content}\" — reframe this as a Bitcoin-related question and ask it back to the user without naming any books or authors."}
            ]
        )
        gpt_response = reframed.choices[0].message.content.strip()
    else:
        # 🧠 Use context to answer normally
        final_prompt = f"""
Use only the context below (derived from Jetking’s internal Bitcoin training material) to answer the user's question.
Do not mention book titles or authors. Do not speculate.

CONTEXT:
{context}

QUESTION:
{req.message.content}

ANSWER:
"""
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

    # ✅ Return full chat history
    history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in history
    ]

    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history
    )

