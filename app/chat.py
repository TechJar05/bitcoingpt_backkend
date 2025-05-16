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

    # ✅ Auto-generate title
    if session.title is None:
        title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
        title_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": title_prompt}]
        )
        generated_title = title_response.choices[0].message.content.strip().strip('"')
        session.title = generated_title[:100]
        db.commit()

    # ✅ Load relevant book context
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=3)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # ✅ JetkingGPT system rules
    system_prompt = """
You are JetkingGPT, a certified AI tutor created to teach and explain the concepts of Bitcoin and related blockchain technologies, based entirely on 10 foundational books provided by Jetking.

You never mention or expose book titles or authors. Do not speculate or hallucinate.

If a user's question is off-topic, reframe it into a Bitcoin-relevant question and ask it back to the user in a friendly tone.

End each valid answer with:
- “Would you like a simpler explanation?”
- “Shall I show you a diagram or example?”
- “Want to go deeper into this topic?”
"""

    # ✅ Rebuild GPT message history (like ChatGPT)
    history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()

    gpt_messages = [{"role": "system", "content": system_prompt}]

    # ✅ Inject context at the start if available
    if context.strip():
        gpt_messages.append({
            "role": "user",
            "content": f"""
Use the context below (from trusted Jetking material) to answer the question. Do not mention book titles or authors.

CONTEXT:
{context}
"""
        })

    # ✅ Add all prior messages
    for msg in history:
        gpt_messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # ✅ Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=gpt_messages
    )
    gpt_response = response.choices[0].message.content.strip()

    # ✅ Save assistant response
    assistant_msg = models.ChatMessage(
        session_id=session.id,
        role="assistant",
        content=gpt_response
    )
    db.add(assistant_msg)
    db.commit()

    # ✅ Return all messages
    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in history
    ]

    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history + [  # include new response
            schemas.MessageOut(role="assistant", content=gpt_response, timestamp=assistant_msg.timestamp)
        ]
    )
