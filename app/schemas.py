from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MessageIn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str]  # null for new chat
    message: MessageIn

class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: datetime

# ✅ New model for YouTube videos
class YouTubeVideo(BaseModel):
    title: str
    url: str

# ✅ Updated response schema
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[MessageOut]
    video_url: Optional[str] = None
    is_fallback: Optional[bool] = False
    youtube_links: List[YouTubeVideo] = [] # ✅ Added field
    
class SessionDetailOut(BaseModel):
    session_id: str
    title: Optional[str]
    created_at: Optional[datetime]
    messages: List[MessageOut]
    saved_videos: Optional[List[YouTubeVideo]] = []
