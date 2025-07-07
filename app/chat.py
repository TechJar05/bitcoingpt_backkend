from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from . import models, schemas
from .database import SessionLocal
            
import random
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from serpapi import GoogleSearch
import os
import re
from typing import List, Dict, Optional

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
serp_api_key = os.getenv("SERPAPI_API_KEY")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# ✅ Enhanced regulation detection with more comprehensive keywords
def is_regulation_query(text: str) -> bool:
    """Detect if the query is about Bitcoin/crypto regulations"""
    regulation_keywords = [
        "regulation", "legal", "illegal", "ban", "banned", "government", "allowed", 
        "prohibited", "policy", "law", "act", "rules", "compliance", "regulatory",
        "sec", "cftc", "finra", "treasury", "sanctions", "kyc", "aml", "tax",
        "legislation", "jurisdiction", "authorized", "licensed", "registered"
    ]
    
    # Check for Bitcoin/crypto context + regulation keywords
    bitcoin_context = ["bitcoin", "crypto", "cryptocurrency", "digital asset", "blockchain"]
    text_lower = text.lower()
    
    has_bitcoin_context = any(keyword in text_lower for keyword in bitcoin_context)
    has_regulation_keyword = any(re.search(rf"\b{word}\b", text_lower) for word in regulation_keywords)
    
    return has_bitcoin_context and has_regulation_keyword

# ✅ Enhanced relevance classification with better prompting
def classify_prompt_relevance(prompt: str) -> str:
    """Classify query relevance with improved accuracy"""
    classification_prompt = f"""
You are a strict domain classifier for a Bitcoin education assistant.

Classify this question into exactly ONE category:

**RELEVANT** - Questions directly about:
- Bitcoin, cryptocurrency, digital assets
- Blockchain technology, mining, wallets
- Crypto trading, exchanges, DeFi
- Bitcoin economics, price analysis
- Crypto security, private keys

**PARTIAL** - Questions that could connect to Bitcoin:
- General finance, investing, economics
- Money, payments, banking systems
- Financial technology, fintech
- Market analysis, trading concepts

**IRRELEVANT** - Questions completely unrelated:
- Geography, politics (without crypto context)
- Sports, entertainment, food, travel
- Health, science, history (without crypto context)
- General technology (without blockchain/crypto)

User question: "{prompt}"

Respond with only one word: relevant, partial, or irrelevant
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # More reliable than gpt-3.5-turbo for classification
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=10,
            temperature=0
        )
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate response
        if classification not in ["relevant", "partial", "irrelevant"]:
            return "irrelevant"  # Default to safe option
            
        return classification
    except Exception as e:
        print(f"Classification error: {e}")
        return "irrelevant"  # Fail safe

def is_short_followup(prompt: str, previous_assistant_msg: str) -> bool:
    """Use GPT to detect if the current prompt is a follow-up (like 'yes', 'how?', 'go on')"""
    check_prompt = f"""
You are an assistant that classifies user inputs.

Given this previous assistant response:
\"\"\"{previous_assistant_msg}\"\"\"

And the user's reply:
\"\"\"{prompt}\"\"\"

Is the user reply a follow-up (e.g., 'yes', 'how?', 'okay', 'continue')? Respond with one word: yes or no
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0,
            max_tokens=2
        )
        return response.choices[0].message.content.strip().lower() == "yes"
    except Exception as e:
        print(f"Follow-up detection error: {e}")
        return False



# ✅ Improved live regulation info fetching
def fetch_live_regulation_info(query: str) -> str:
    """Fetch live regulation information with better error handling"""
    if not serp_api_key:
        return "Live regulation data unavailable. Please check official government sources."
    
    try:
        # Enhance query for better results
        enhanced_query = f"Bitcoin cryptocurrency {query} regulation 2024 2025"
        
        params = {
            "q": enhanced_query,
            "api_key": serp_api_key,
            "gl": "in",
            "num": 5  # Get more results for better quality
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = []

        for result in results.get("organic_results", []):
            if "snippet" in result:
                snippets.append(result["snippet"])
            if len(snippets) >= 3:
                break

        if snippets:
            return "\n\n".join(snippets)
        else:
            return "No current regulation information found. Please consult official regulatory websites."
            
    except Exception as e:
        print(f"Live data fetch error: {e}")
        return "Unable to fetch live regulation data. Please check official sources."

def fetch_youtube_videos(query: str, relevance: str = "relevant", context: str = "", max_results: int = 2) -> List[Dict[str, str]]:
    """Fetch YouTube videos using dynamic keyword expansion for partial prompts."""
    if not serp_api_key:
        return []

    try:
        # Step 1: Extract key terms using GPT if partial
        if relevance == "partial":
            prompt = f"""
You're helping improve YouTube search results for educational purposes.

Given this user question:
"{query}"

And the educational context (about Bitcoin, crypto, and blockchain):
"{context[:1000]}"  # limit tokens

List 3–5 related keywords or short phrases that link the question to Bitcoin/crypto concepts. Separate them by commas.
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.3
            )
            extracted = response.choices[0].message.content.strip()
            keywords = ", ".join([kw.strip() for kw in extracted.split(",") if kw.strip()])
            search_query = f"{query} {keywords}"
        else:
            search_query = f"Bitcoin {query} tutorial explanation"

        # Step 2: Run SerpAPI YouTube search
        params = {
            "engine": "youtube",
            "search_query": search_query,
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
    except Exception as e:
        print(f"YouTube fetch error: {e}")
        return []



# ✅ Context relevance scoring
def score_context_relevance(context: str, query: str) -> float:
    """Score how relevant the retrieved context is to the query"""
    if not context or not query:
        return 0.0
    
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    # Calculate word overlap
    overlap = len(query_words.intersection(context_words))
    total_query_words = len(query_words)
    
    return overlap / total_query_words if total_query_words > 0 else 0.0

# ✅ Enhanced response generation based on relevance
def generate_contextual_response(relevance: str, context: str, query: str, is_regulation: bool) -> str:
    """Generate appropriate system prompt based on relevance level"""

    base_prompt = """You are JetkingGPT — an expert Bitcoin education assistant created by Jetking, a leader in digital skills education.

Formatting Rules:
- Use Byte-Sized Learning: keep answers brief and digestible
- Avoid emojis or emoticons
- Use short paragraphs (2–3 lines max)
- Use bullet points where helpful
- Simplify complex ideas using examples and analogies
"""

    if relevance == "relevant":
        if is_regulation:
            return f"""{base_prompt}

You specialize in Bitcoin and cryptocurrency topics. For regulation questions, combine your trained knowledge with the live data provided. Make it clear when data is real-time.

Style Guidelines:
- Focus on practical implications of the regulation
- Break down legal language simply
- Clarify if rules differ by region
"""
        else:
            return f"""{base_prompt}

You answer Bitcoin questions using Jetking's 10-book training. Stay focused on Bitcoin, cryptocurrency, and blockchain concepts.

Style Guidelines:
- Define key terms simply
- Prioritize clarity over depth when in doubt
- Engage the user like a tutor would in a classroom
"""
    
    elif relevance == "partial":
        return f"""{base_prompt}

The user’s question is finance-related. Your job is to draw meaningful connections to Bitcoin where appropriate.

Style Guidelines:
- Don’t force a connection if it’s not useful
- Explain Bitcoin’s relevance in the context
- Guide the user to explore more specific Bitcoin topics if needed
"""

    return ""


@router.post("/chat/", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest, db: Session = Depends(get_db)):
    """Enhanced chat endpoint with better relevance handling"""
    
    # 🔍 Classify prompt relevance with validation
    # ✅ Determine relevance, and detect if it's a follow-up continuation
    session_obj = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
    previous_msg = None
    if session_obj:
        previous_msg = (
            db.query(models.ChatMessage)
            .filter_by(session_id=session_obj.id, role="assistant")
            .order_by(models.ChatMessage.timestamp.desc())
            .first()
        )


    try:
        if previous_msg and is_short_followup(req.message.content, previous_msg.content):
            relevance = "relevant"  # Treat follow-up as continuation
            req.message.content = f"{previous_msg.content.strip()}\n\nUser's follow-up: {req.message.content.strip()}"
            print("Follow-up detected — chaining context")
        else:
            relevance = classify_prompt_relevance(req.message.content)
    except Exception as e:
        print(f"Classification failed: {e}")
        relevance = "irrelevant"

    
    # ❌ Block irrelevant queries with helpful message
    if relevance == "irrelevant":
        return schemas.ChatResponse(
            session_id=req.session_id or str(uuid4()),
            reply="""I'm specialized in Bitcoin and cryptocurrency education. I can help you with:

• Bitcoin basics and how it works
• Blockchain technology and mining
• Cryptocurrency wallets and security
• Bitcoin trading and market analysis
• Crypto regulations and compliance
• Digital asset management

Please ask a question related to Bitcoin or cryptocurrency topics!""",
            history=[],
            video_url=None,
            is_fallback=True,
            youtube_links=[]
        )

    # ✅ Session management
    try:
        if req.session_id:
            session = db.query(models.ChatSession).filter_by(session_id=req.session_id).first()
            if not session:
                session = models.ChatSession(session_id=req.session_id)
                db.add(session)
                db.commit()
        else:
            session = models.ChatSession(session_id=str(uuid4()))
            db.add(session)
            db.commit()
            db.refresh(session)
    except Exception as e:
        print(f"Session error: {e}")
        return schemas.ChatResponse(
            session_id=str(uuid4()),
            reply="I encountered an error processing your session. Please try again.",
            history=[],
            video_url=None,
            is_fallback=True,
            youtube_links=[]
        )

    # ✅ Save user message
    try:
        user_msg = models.ChatMessage(
            session_id=session.id,
            role="user",
            content=req.message.content
        )
        db.add(user_msg)
        db.commit()
    except Exception as e:
        print(f"Message save error: {e}")

    # ✅ Generate session title if needed
    if session.title is None:
        try:
            title_prompt = f"Create a concise 4-6 word title for this Bitcoin-related query: \"{req.message.content}\""
            title_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": title_prompt}],
                max_tokens=20
            )
            session.title = title_response.choices[0].message.content.strip().strip('"')[:100]
            db.commit()
        except Exception as e:
            print(f"Title generation error: {e}")
            session.title = "Bitcoin Chat Session"
            db.commit()

    # ✅ Load and score context relevance
    try:
        vectorstore = FAISS.load_local("vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(req.message.content, k=3)  # Get more candidates
        
        if docs:
            # Score and filter context
            scored_docs = []
            for doc in docs:
                score = score_context_relevance(doc.page_content, req.message.content)
                if score > 0.1:  # Minimum relevance threshold
                    scored_docs.append((doc, score))
            
            # Sort by relevance score and take top 2
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            context = "\n\n".join([doc[0].page_content for doc in scored_docs[:2]])
        else:
            context = ""
    except Exception as e:
        print(f"Vector search error: {e}")
        context = ""

    # ✅ Load video context
    try:
        video_vectorstore = FAISS.load_local("video_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        video_docs = video_vectorstore.similarity_search(req.message.content, k=1)
        
        fallback_video_url = "https://vimeo.com/1086262917/e341ef910d"
        video_url = video_docs[0].metadata.get("video_url", fallback_video_url) if video_docs else fallback_video_url
        is_fallback = not video_docs or video_url == fallback_video_url
    except Exception as e:
        print(f"Video search error: {e}")
        video_url = "https://vimeo.com/1086262917/e341ef910d"
        is_fallback = True

    # ✅ Fetch additional resources based on relevance
    # ✅ Intelligent video reuse
    
    import numpy as np

    # ✅ YouTube video generation control based on prompt similarity
    youtube_links = []
    should_generate_video = False
    embedding_model = OpenAIEmbeddings()
    new_prompt_embedding = embedding_model.embed_query(req.message.content)

    # Compute cosine similarity with previous prompt embedding if exists
    def cosine_similarity(vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarity_score = 0.0
    if session.last_prompt_embedding:
        similarity_score = cosine_similarity(new_prompt_embedding, session.last_prompt_embedding)

    # Threshold: if the prompt has changed enough, allow video regeneration
    if relevance in ["relevant", "partial"]:
        if similarity_score < 0.9:
            should_generate_video = True

    # Decide whether to reuse or fetch new
    # Decide whether to reuse or fetch new
    if session.saved_videos and not should_generate_video:
        youtube_links = session.saved_videos
        is_fallback = False
    else:
        youtube_links = fetch_youtube_videos(req.message.content, relevance, context)
        if youtube_links:
            # Combine old + new, avoid duplicates
            existing_links = session.saved_videos or []
            new_unique_links = [v for v in youtube_links if v not in existing_links]

            session.saved_videos = existing_links + new_unique_links
            session.last_prompt_embedding = new_prompt_embedding
            db.commit()





    # ✅ Check for regulation query
    is_regulation = is_regulation_query(req.message.content)
    
    # ✅ Generate system prompt based on context
    system_prompt = generate_contextual_response(relevance, context, req.message.content, is_regulation)

    # ✅ Get chat history
    try:
        full_history = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
        limited_history = full_history[-8:]  # Increased context window
    except Exception as e:
        print(f"History fetch error: {e}")
        limited_history = []

    # ✅ Build final prompt based on query type and relevance
    if is_regulation and relevance == "relevant":
        live_data = fetch_live_regulation_info(req.message.content)
        final_prompt = f"""REGULATION QUERY: "{req.message.content}"

Use this live regulatory information to provide an accurate, up-to-date response:

CURRENT REGULATION DATA:
{live_data}

Additional context from our Bitcoin curriculum:
{context}

Provide a comprehensive answer that combines live regulatory information with educational context."""

    elif relevance == "relevant":
        final_prompt = f"""BITCOIN QUERY: "{req.message.content}"

Use this context from Jetking's Bitcoin curriculum to provide a comprehensive answer:

CONTEXT:
{context}

Focus on educational value and practical understanding."""

    elif relevance == "partial":
        final_prompt = f"""FINANCE-RELATED QUERY: "{req.message.content}"

This question relates to finance/economics. Connect it to Bitcoin concepts where appropriate.

Bitcoin context:
{context}

Explain how Bitcoin fits into this broader financial concept, but don't force unnatural connections."""

    
    # ✅ Prepare GPT messages with full chat history (like ChatGPT)
    # Build GPT messages using full context
    gpt_messages = [{"role": "system", "content": system_prompt}]
    for msg in limited_history:
        gpt_messages.append({"role": msg.role, "content": msg.content})

    # Detect if user input is a follow-up (short, vague)
    if len(req.message.content.strip()) <= 6 and req.message.content.lower() in ["yes", "sure", "okay", "ok", "go on", "continue"]:
        # Find the most recent assistant message
        last_assistant_msg = next((m for m in reversed(full_history) if m.role == "assistant"), None)
        if last_assistant_msg:
            gpt_messages.append({"role": "assistant", "content": last_assistant_msg.content})
            gpt_messages.append({"role": "user", "content": req.message.content.strip()})
        else:
            # fallback if no assistant message
            gpt_messages.append({"role": "user", "content": final_prompt})
    else:
        gpt_messages.append({"role": "user", "content": final_prompt})




    # ✅ Generate response with error handling
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Better model for educational responses
            messages=gpt_messages,
            temperature=0.7,
            max_tokens=700
        )
        gpt_response = response.choices[0].message.content.strip()
        
        # ✅ Generate follow-up question to keep user engaged
        try:
            followup_prompt = f"""
        Given the user question: "{req.message.content}"

        Suggest a simple follow-up question that builds on this topic to keep learning flowing. Keep it short and educational. Only return the follow-up question, no preamble.
        """
            followup_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": followup_prompt}],
                max_tokens=50,
                temperature=0.7
            )
            followup_question = followup_response.choices[0].message.content.strip()
            

            leads = [
                "**Follow-up to explore:**",
                "**To deepen your understanding:**",
                "**You might also explore:**",
                "**Next question to consider:**"
            ]
            lead_in = random.choice(leads)
            gpt_response += f"\n\n{lead_in} {followup_question}"

            
        except Exception as e:
            print(f"Follow-up generation error: {e}")

        
        # Add context note for partial relevance with YouTube suggestion
        if relevance == "partial":
            youtube_note = ""
            if youtube_links:
                youtube_note = f"\n\n📺 **Watch these videos to learn how {req.message.content.lower()} relates to Bitcoin:**"
                for i, video in enumerate(youtube_links, 1):
                    youtube_note += f"\n{i}. [{video['title']}]({video['url']})"
            
            gpt_response += f"{youtube_note}\n\n*Note: I've connected this topic to Bitcoin concepts. For more specific Bitcoin questions, feel free to ask!*"
            
    except Exception as e:
        print(f"GPT response error: {e}")
        gpt_response = "I'm experiencing technical difficulties. Please try rephrasing your Bitcoin-related question."
    
    
    gpt_response = remove_emojis(gpt_response)
    # ✅ Save assistant response
    try:
        assistant_msg = models.ChatMessage(
            session_id=session.id,
            role="assistant",
            content=gpt_response
        )
        db.add(assistant_msg)
        db.commit()
        
        # Update history
        full_history.append(assistant_msg)
    except Exception as e:
        print(f"Response save error: {e}")

    # ✅ Format response history
    formatted_history = [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in full_history
    ]

    
    return schemas.ChatResponse(
        session_id=session.session_id,
        reply=gpt_response,
        history=formatted_history,
        video_url=video_url,
        is_fallback=is_fallback,
        youtube_links=youtube_links
    )