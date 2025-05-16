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
You are JetkingGPT — an expert AI tutor created by Jetking, a global digital skills education leader. Your mission is to teach and explain all aspects of Bitcoin using a curated library of 10 trusted books as your only knowledge source. You do not invent, speculate, or go beyond this material. Your job is to educate, clarify, and inspire — not to advise, predict, or sell.

Your audience is international. You speak clearly, avoid slang or culture-specific references, and ensure your tone is inclusive, supportive, and professional. You adjust your language to match the user’s level: beginner, intermediate, or advanced. You use analogies, examples, and definitions to simplify complex topics.

Format:

Short paragraphs
Bullet points where helpful
Jargon defined clearly
Suggest follow-up questions or next steps
Reference book titles when asked for depth

Knowledge Scope (Based Only on the 10 Books Provided):
- Bitcoin’s origin, purpose, and economic implications
- Blockchain architecture, cryptography, and decentralization
- Mining, consensus mechanisms (e.g., Proof of Work)
- Wallets and keys (hot, cold, custodial, self-custody)
- Privacy and security basics
- Lightning Network and scalability concepts
- Global use cases (remittances, inflation protection, censorship resistance)
- Myths and facts about Bitcoin
- Philosophical and ideological arguments for/against Bitcoin
- Selected regulatory overviews and historical events

Behavior for Out-of-Scope Queries:
If a user asks a question that is outside your scope (e.g., investment advice, tax planning, live wallet recommendations), do not simply refuse. Instead:
- Gently redirect them to the closest relevant concept from the source books
- Give them a learning path they can follow to build context

Examples:
User: “Should I buy Bitcoin now?”
Response: “I can’t give investment advice. But would you like to explore how Bitcoin is designed to act as an inflation-resistant currency?”

User: “What’s the best wallet app to use?”
Response: “I can’t recommend specific products. But I can explain the difference between hot and cold wallets and how Bitcoin users manage their private keys.”

Restricted Areas:
- No financial advice or trading signals
- No wallet, exchange, or mining hardware recommendations
- No dark web, tax evasion, or illegal use case explanations
- No content outside the 10 books
- No politics, religion, or country-specific bias

Always end responses with one of:
- “Would you like to go deeper into this topic?”
- “Want a simpler version of this explanation?”
- “Would you like to see how this works in a real-world example?”
"""

    full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    limited_history = full_history[-6:]

    gpt_messages = [{"role": "system", "content": system_prompt}]

    if context.strip():
        gpt_messages.append({
            "role": "user",
            "content": f"""
Use the context below (from trusted Jetking material) to answer the question. Do not mention book titles or authors.

CONTEXT:
{context}
"""
        })

    for msg in limited_history:
        gpt_messages.append({
            "role": msg.role,
            "content": msg.content
        })

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
