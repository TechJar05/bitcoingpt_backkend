# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from uuid import uuid4
# from . import models, schemas
# from .database import SessionLocal

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI
# from serpapi import GoogleSearch
# import os
# import re

# router = APIRouter()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# serp_api_key = os.getenv("SERPAPI_API_KEY")

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # ✅ Simple keyword check
# def is_regulation_query(text):
#     keywords = ["regulation", "legal", "illegal", "ban", "banned", "government", "allowed", "prohibited", "policy", "law", "act", "rules"]
#     return any(re.search(rf"\b{word}\b", text.lower()) for word in keywords)

# # ✅ Fetch real-time data via SerpAPI
# def fetch_live_regulation_info(query):
#     params = {
#         "q": query,
#         "api_key": serp_api_key,
#         "gl": "in",  # You can customize this by user location or request
#     }
#     search = GoogleSearch(params)
#     results = search.get_dict()
#     snippets = []

#     for result in results.get("organic_results", []):
#         if "snippet" in result:
#             snippets.append(result["snippet"])
#         if len(snippets) >= 3:
#             break

#     return "\n\n".join(snippets) if snippets else "No live data found. Please check an official source."

# @router.post("/chat/", response_model=schemas.ChatResponse)
# def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
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

#     if session.title is None:
#         title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
#         title_response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": title_prompt}]
#         )
#         session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
#         db.commit()

#     # Load vector index for book context
#     vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     docs = vectorstore.similarity_search(req.message.content, k=2)
#     context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

#     # Load video index with fallback
#     video_vectorstore = FAISS.load_local("video_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     fallback_video_url = "https://vimeo.com/1086262917/e341ef910d"
#     video_docs = video_vectorstore.similarity_search(req.message.content, k=1)
#     video_url = video_docs[0].metadata["video_url"] if video_docs else fallback_video_url
#     is_fallback = not video_docs or video_docs[0].metadata["video_url"] == fallback_video_url

#     # Default system prompt
#     system_prompt = """
# You are JetkingGPT — an expert AI tutor created by Jetking, a global leader in digital skills education. You explain all aspects of Bitcoin using 10 trusted books. If the topic is Bitcoin regulation, use real-time sources. For all other topics, respond using the 10 books only.

# Teaching Style:
# - Clear short paragraphs
# - Bullet points where useful
# - Analogies, examples, definitions
# - Tailored by user level (beginner, teen, expert)
# """

#     # Gather recent messages
#     full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
#     limited_history = full_history[-6:]

#     # Use live data for regulation queries
#     if is_regulation_query(req.message.content):
#         live_data = fetch_live_regulation_info(req.message.content)
#         final_prompt = f"""The user asked: "{req.message.content}"

# Use the live data below to answer in a student-friendly tone. Clarify it's real-time info.

# LIVE DATA:
# {live_data}

# Answer:"""
#     else:
#         final_prompt = f"""
# Use the context below (from Jetking’s Bitcoin books) to answer the user's question. Do not mention book titles.

# CONTEXT:
# {context}

# QUESTION:
# {req.message.content}
# """

#     gpt_messages = [{"role": "system", "content": system_prompt}]
#     for msg in limited_history:
#         gpt_messages.append({"role": msg.role, "content": msg.content})
#     gpt_messages.append({"role": "user", "content": final_prompt})

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=gpt_messages
#     )
#     gpt_response = response.choices[0].message.content.strip()

#     # Save assistant response
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
#         history=formatted_history,
#         video_url=video_url,
#         is_fallback=is_fallback
#     )


from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from . import models, schemas
from .database import SessionLocal

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from serpapi import GoogleSearch
import os
import re

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
serp_api_key = os.getenv("SERPAPI_API_KEY")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Detect regulation questions
def is_regulation_query(text):
    keywords = ["regulation", "legal", "illegal", "ban", "banned", "government", "allowed", "prohibited", "policy", "law", "act", "rules"]
    return any(re.search(rf"\b{word}\b", text.lower()) for word in keywords)

# ✅ Get live info for regulation-related queries
def fetch_live_regulation_info(query):
    params = {
        "q": query,
        "api_key": serp_api_key,
        "gl": "in",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = []

    for result in results.get("organic_results", []):
        if "snippet" in result:
            snippets.append(result["snippet"])
        if len(snippets) >= 3:
            break

    return "\n\n".join(snippets) if snippets else "No live data found. Please check an official source."

# ✅ Fetch YouTube videos via SerpAPI
def fetch_youtube_videos(query, max_results=2):
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": serp_api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    videos = []
    for video in results.get("video_results", [])[:max_results]:
        if "link" in video and "title" in video:
            videos.append({
                "title": video["title"],
                "url": video["link"]
            })
    return videos

@router.post("/chat/", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
    if req.session_id:
        session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
    else:
        session = models.ChatSession(session_id=str(uuid4()))
        db.add(session)
        db.commit()
        db.refresh(session)

    # Save user message
    user_msg = models.ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message.content
    )
    db.add(user_msg)
    db.commit()

    # Auto-generate topic title if not already set
    if session.title is None:
        title_prompt = f"Generate a short, clear topic title (4–6 words) for this message:\n\n\"{req.message.content}\""
        title_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": title_prompt}]
        )
        session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
        db.commit()

    # Load book context vectorstore
    vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(req.message.content, k=2)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Load video vector index
    video_vectorstore = FAISS.load_local("video_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    fallback_video_url = "https://vimeo.com/1086262917/e341ef910d"
    video_docs = video_vectorstore.similarity_search(req.message.content, k=1)
    video_url = video_docs[0].metadata["video_url"] if video_docs else fallback_video_url
    is_fallback = not video_docs or video_docs[0].metadata["video_url"] == fallback_video_url

    # ✅ Fetch relevant YouTube videos
    youtube_links = fetch_youtube_videos(req.message.content)

    # System prompt for GPT
    system_prompt = """
You are JetkingGPT — an expert AI tutor created by Jetking, a global leader in digital skills education. You explain all aspects of Bitcoin using 10 trusted books. If the topic is Bitcoin regulation, use real-time sources. For all other topics, respond using the 10 books only.

Teaching Style:
- Clear short paragraphs
- Bullet points where useful
- Analogies, examples, definitions
- Tailored by user level (beginner, teen, expert)
"""

    # Recent chat history
    full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    limited_history = full_history[-6:]

    # Compose GPT prompt
    if is_regulation_query(req.message.content):
        live_data = fetch_live_regulation_info(req.message.content)
        final_prompt = f"""The user asked: "{req.message.content}"

Use the live data below to answer in a student-friendly tone. Clarify it's real-time info.

LIVE DATA:
{live_data}

Answer:"""
    else:
        final_prompt = f"""
Use the context below (from Jetking’s Bitcoin books) to answer the user's question. Do not mention book titles.

CONTEXT:
{context}

QUESTION:
{req.message.content}
"""

    gpt_messages = [{"role": "system", "content": system_prompt}]
    for msg in limited_history:
        gpt_messages.append({"role": msg.role, "content": msg.content})
    gpt_messages.append({"role": "user", "content": final_prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages
    )
    gpt_response = response.choices[0].message.content.strip()

    # Save assistant response
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
        video_url=video_url,
        is_fallback=is_fallback,
        youtube_links=youtube_links  # ✅ New YouTube titles + links
    )

