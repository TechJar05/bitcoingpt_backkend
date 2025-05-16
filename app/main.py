from fastapi import FastAPI
from .models import Base
from .database import engine
from .chat import router as chat_router
from .session import router as session_router

app = FastAPI()

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Register API routes
app.include_router(chat_router)
app.include_router(session_router)
