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

# Behavior for Out-of-Scope Queries:
# If a user asks something unrelated to Bitcoin, always redirect them with this response:
# "I’m here to help you understand Bitcoin using trusted resources. Would you like to ask something about Bitcoin?"

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
#         gpt_response = "I’m here to help you understand Bitcoin using trusted resources. Would you like to ask something about Bitcoin?"

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

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from . import models, schemas
from .database import SessionLocal

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
import re

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def detect_regulation_query(text: str) -> bool:
    return any(kw in text.lower() for kw in ["legal", "ban", "regulation", "tax", "government", "allowed", "country"])

def detect_childlike_query(text: str) -> str:
    if any(kw in text.lower() for kw in ["kid", "5 year old", "cartoon", "baby", "like a child"]):
        return "child"
    elif any(kw in text.lower() for kw in ["teen", "school", "fun way", "easy way", "student"]):
        return "teen"
    return "default"

@router.post("/chat/", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
    if req.session_id:
        session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
    else:
        session = models.ChatSession(session_id=str(uuid4()))
        db.add(session)
        db.commit()
        db.refresh(session)

    user_msg = models.ChatMessage(session_id=session.id, role="user", content=req.message.content)
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

    # Load book vector index
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=2)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Load video vector index with fallback
    video_vectorstore = FAISS.load_local("video_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    fallback_video_url = "https://vimeo.com/1086262917/e341ef910d"
    video_docs = video_vectorstore.similarity_search(req.message.content, k=1)
    video_url = video_docs[0].metadata["video_url"] if video_docs else fallback_video_url
    is_fallback = not video_docs or video_docs[0].metadata["video_url"] == fallback_video_url

    # Teaching style detection
    audience_mode = detect_childlike_query(req.message.content)

    tone_prefix = ""
    if audience_mode == "child":
        tone_prefix = "Please explain this like I’m a 5-year-old using playful examples or metaphors.\n"
    elif audience_mode == "teen":
        tone_prefix = "Explain this in a relatable, teenage-friendly way with modern analogies.\n"

    # GPT prompt construction
    full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    limited_history = full_history[-6:]

    gpt_messages = [{"role": "system", "content": "You are JetkingGPT, a helpful AI tutor who only explains Bitcoin using books or reliable public knowledge when needed."}]

    if detect_regulation_query(req.message.content):
        gpt_messages.append({
            "role": "user",
            "content": tone_prefix + "This may be a regulatory/legal question. If helpful, include updated public info from your model knowledge (e.g. about India, US, or global laws). Question: " + req.message.content
        })
    elif not context.strip():
        gpt_messages.append({
            "role": "user",
            "content": tone_prefix + "This topic was not found in the books. Please respond based on general understanding of Bitcoin but keep it neutral and factual. Question: " + req.message.content
        })
    else:
        gpt_messages.append({
            "role": "user",
            "content": tone_prefix + f"Answer using the context below only:\n\n{context}"
        })
        for msg in limited_history:
            gpt_messages.append({"role": msg.role, "content": msg.content})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages
    )
    gpt_response = response.choices[0].message.content.strip()

    assistant_msg = models.ChatMessage(session_id=session.id, role="assistant", content=gpt_response)
    db.add(assistant_msg)
    db.commit()

    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in full_history
    ] + [schemas.MessageOut(role="assistant", content=gpt_response, timestamp=assistant_msg.timestamp)]

    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history,
        video_url=video_url,
        is_fallback=is_fallback
    )
