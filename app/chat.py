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

    # Load book vector index for context
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=2)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Load video vector index and fetch top 3 results
    video_vectorstore = FAISS.load_local("video_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    video_docs = video_vectorstore.similarity_search(req.message.content, k=3)
    video_urls = [doc.metadata["video_url"] for doc in video_docs if "video_url" in doc.metadata]

    # Fallback to a default video if none matched
    if not video_urls:
        video_urls = ["https://vimeo.com/1086262917/e341ef910d"]  # Default: What is Bitcoin
        is_fallback = True
    else:
        is_fallback = False

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
        history=formatted_history,
        video_urls=video_urls,
        is_fallback=is_fallback
    )

