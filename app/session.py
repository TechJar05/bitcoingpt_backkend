from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import models, schemas
from typing import List
from uuid import uuid4
from pydantic import BaseModel

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Get all sessions for sidebar
@router.get("/sessions/", response_model=List[dict])
def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(models.ChatSession).order_by(models.ChatSession.created_at.desc()).all()
    output = []
    for session in sessions:
        first_msg = db.query(models.ChatMessage).filter_by(session_id=session.id, role="user").order_by(models.ChatMessage.timestamp.asc()).first()
        output.append({
            "session_id": session.session_id,
            "created_at": session.created_at,
            "title": session.title or (first_msg.content[:60] + "..." if first_msg else "Untitled session")
        })
    return output

# ✅ Get full chat history by session ID
@router.get("/sessions/{session_id}/", response_model=List[schemas.MessageOut])
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.query(models.ChatMessage).filter_by(session_id=session.id).order_by(models.ChatMessage.timestamp).all()
    return [
        schemas.MessageOut(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in messages
    ]

# ✅ Create a new empty session
@router.post("/sessions/new", status_code=status.HTTP_201_CREATED)
def create_new_session(db: Session = Depends(get_db)):
    new_session = models.ChatSession(session_id=str(uuid4()))
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"session_id": new_session.session_id}

# ✅ Rename session title
class RenameRequest(BaseModel):
    new_title: str

@router.put("/sessions/{session_id}/rename")
def rename_session(session_id: str, payload: RenameRequest, db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.title = payload.new_title.strip()
    db.commit()
    return {"message": "Session renamed successfully"}

# ✅ Delete a session
@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(models.ChatSession).filter_by(session_id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete all related messages
    db.query(models.ChatMessage).filter_by(session_id=session.id).delete()

    # Delete the session
    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}
