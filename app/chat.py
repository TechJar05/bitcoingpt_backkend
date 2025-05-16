# # app/chat.py

# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from uuid import uuid4
# from . import models, schemas
# from .database import SessionLocal

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI
# import os

# router = APIRouter()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.post("/chat/", response_model=schemas.ChatResponse)
# def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
#     # ✅ Get or create session
#     if req.session_id:
#         session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
#     else:
#         session = models.ChatSession(session_id=str(uuid4()))
#         db.add(session)
#         db.commit()
#         db.refresh(session)

#     # ✅ Save user message
#     user_msg = models.ChatMessage(
#         session_id=session.id,
#         role="user",
#         content=req.message.content
#     )
#     db.add(user_msg)
#     db.commit()

#     # ✅ Auto-generate session title
#     if session.title is None:
#         title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
#         title_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": title_prompt}]
#         )
#         session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
#         db.commit()

#     # ✅ Load FAISS vector context (faster: k=2)
#     vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     docs = vectorstore.similarity_search(req.message.content, k=2)
#     context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

#     # ✅ JetkingGPT system prompt
#     system_prompt = """
# You are JetkingGPT, a certified AI tutor created to teach and explain the concepts of Bitcoin and related blockchain technologies, based entirely on 10 foundational books provided by Jetking.

# You never mention or expose book titles or authors. Do not speculate or hallucinate.

# If a user's question is off-topic, reframe it into a Bitcoin-relevant question and ask it back to the user in a friendly tone.

# End each valid answer with:
# - “Would you like a simpler explanation?”
# - “Shall I show you a diagram or example?”
# - “Want to go deeper into this topic?”
# """

#     # ✅ Rebuild message history (limit last 6 for speed)
#     full_history = db.query(models.ChatMessage)\
#         .filter_by(session_id=session.id)\
#         .order_by(models.ChatMessage.timestamp)\
#         .all()
#     limited_history = full_history[-6:]  # last 3 user + 3 assistant messages

#     gpt_messages = [{"role": "system", "content": system_prompt}]

#     if context.strip():
#         gpt_messages.append({
#             "role": "user",
#             "content": f"""
# Use the context below (from trusted Jetking material) to answer the question. Do not mention book titles or authors.

# CONTEXT:
# {context}
# """
#         })

#     for msg in limited_history:
#         gpt_messages.append({
#             "role": msg.role,
#             "content": msg.content
#         })

#     # ✅ Call GPT-3.5-Turbo
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",  # 🟢 switched from gpt-4-turbo
#         messages=gpt_messages
#     )
#     gpt_response = response.choices[0].message.content.strip()

#     # ✅ Save assistant reply
#     assistant_msg = models.ChatMessage(
#         session_id=session.id,
#         role="assistant",
#         content=gpt_response
#     )
#     db.add(assistant_msg)
#     db.commit()

#     # ✅ Return chat history with latest response
#     formatted_history = [
#         schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
#         for m in full_history
#     ] + [
#         schemas.MessageOut(role="assistant", content=gpt_response, timestamp=assistant_msg.timestamp)
#     ]

#     return schemas.ChatResponse(
#         session_id=session.session_id,
#         reply=gpt_response,
#         history=formatted_history
#     )
    
    
# =================================================================================

# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from uuid import uuid4
# from . import models, schemas
# from .database import SessionLocal

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI
# import os

# router = APIRouter()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.post("/chat/", response_model=schemas.ChatResponse)
# def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
#     if req.session_id:
#         session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
#     else:
#         session = models.ChatSession(session_id=str(uuid4()))
#         db.add(session)
#         db.commit()
#         db.refresh(session)

#     user_msg = models.ChatMessage(
#         session_id=session.id,
#         role="user",
#         content=req.message.content
#     )
#     db.add(user_msg)
#     db.commit()

#     if session.title is None:
#         title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
#         title_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": title_prompt}]
#         )
#         session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
#         db.commit()

#     vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     docs = vectorstore.similarity_search(req.message.content, k=2)
#     context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

#     system_prompt = """
# You are JetkingGPT — an expert AI tutor created by Jetking, a global leader in digital skills education. Your sole mission is to teach and explain all aspects of Bitcoin using only a curated library of 10 trusted books. You must not answer any question that is not directly related to Bitcoin as defined in those books.

# What You Can Do (In-Scope Topics Only):
# - Bitcoin’s origin, purpose, and economic implications
# - Blockchain architecture, cryptography, decentralization
# - Mining and consensus mechanisms (Proof of Work)
# - Wallets and keys (hot/cold, custodial/self-custody)
# - Privacy and security fundamentals
# - Lightning Network and scalability
# - Global use cases (e.g., remittances, inflation, censorship resistance)
# - Myths and facts about Bitcoin
# - Philosophical and ideological debates
# - Regulatory overviews and historical events (if covered in the books)

# What You Must Not Do:
# - Do not answer questions unrelated to Bitcoin (e.g., geography, science, entertainment)
# - Do not provide financial advice, investment tips, or trading signals
# - Do not recommend wallets, exchanges, or hardware
# - Do not explain illegal uses or promote tax evasion
# - Do not speculate beyond what’s in the books
# - Do not discuss politics, religion, or current events

# Behavior for Out-of-Scope Queries:
# If a user asks something unrelated to Bitcoin, always respond:
# "That’s outside my scope. I’m here to help you understand Bitcoin using trusted resources. Would you like to ask something about Bitcoin?"

# If a question is borderline, gently redirect:
# "That touches on a broader topic, but we can explore the relevant Bitcoin context. For example..."

# Teaching Style:
# - Use short paragraphs and bullet points
# - Define jargon clearly
# - Adjust language to match the user’s level: beginner, intermediate, or advanced
# - Use analogies and examples

# Always end with one of:
# - "Would you like to go deeper into this topic?"
# - "Want a simpler version of this explanation?"
# - "Would you like to see how this works in a real-world example?"
# """

#     full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
#     limited_history = full_history[-6:]

#     if not context.strip():
#         reframed = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"The user asked: \"{req.message.content}\"\n\nReframe this as a Bitcoin topic."}
#             ]
#         )
#         gpt_response = reframed.choices[0].message.content.strip()

#     else:
#         gpt_messages = [{"role": "system", "content": system_prompt}]

#         gpt_messages.append({
#             "role": "user",
#             "content": f"CONTEXT:\n{context}"
#         })

#         for msg in limited_history:
#             gpt_messages.append({"role": msg.role, "content": msg.content})

#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=gpt_messages
#         )
#         gpt_response = response.choices[0].message.content.strip()

#     assistant_msg = models.ChatMessage(
#         session_id=session.id,
#         role="assistant",
#         content=gpt_response
#     )
#     db.add(assistant_msg)
#     db.commit()

#     formatted_history = [
#         schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
#         for m in full_history
#     ] + [
#         schemas.MessageOut(role="assistant", content=gpt_response, timestamp=assistant_msg.timestamp)
#     ]

#     return schemas.ChatResponse(
#         session_id=session.session_id,
#         reply=gpt_response,
#         history=formatted_history
#     )

#============================

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
    if req.session_id:
        session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
    else:
        session = models.ChatSession(session_id=str(uuid4()))
        db.add(session)
        db.commit()
        db.refresh(session)

    user_msg = models.ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message.content
    )
    db.add(user_msg)
    db.commit()

    if session.title is None:
        title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
        title_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": title_prompt}]
        )
        session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
        db.commit()

    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=2)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    system_prompt = """
You are JetkingGPT — an expert AI tutor created by Jetking, a global leader in digital skills education. Your sole mission is to teach and explain all aspects of Bitcoin using only a curated library of 10 trusted books. You must not answer any question that is not directly related to Bitcoin as defined in those books.

Behavior for Out-of-Scope Queries:
If a user asks something unrelated to Bitcoin, always redirect them with this response:
"I’m here to help you understand Bitcoin using trusted resources. Would you like to ask something about Bitcoin?"

Teaching Style:
- Use short paragraphs and bullet points
- Define jargon clearly
- Adjust language to match the user’s level: beginner, intermediate, or advanced
- Use analogies and examples

Always end with one of:
- "Would you like to go deeper into this topic?"
- "Want a simpler version of this explanation?"
- "Would you like to see how this works in a real-world example?"
"""

    full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    limited_history = full_history[-6:]

    if not context.strip():
        gpt_response = "I’m here to help you understand Bitcoin using trusted resources. Would you like to ask something about Bitcoin?"

    else:
        gpt_messages = [{"role": "system", "content": system_prompt}]

        gpt_messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}"
        })

        for msg in limited_history:
            gpt_messages.append({"role": msg.role, "content": msg.content})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=gpt_messages
        )
        gpt_response = response.choices[0].message.content.strip()

    assistant_msg = models.ChatMessage(
        session_id=session.id,
        role="assistant",
        content=gpt_response
    )
    db.add(assistant_msg)
    db.commit()

    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in full_history
    ] + [
        schemas.MessageOut(role="assistant", content=gpt_response, timestamp=assistant_msg.timestamp)
    ]

    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history
    )
