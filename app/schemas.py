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

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[MessageOut]
