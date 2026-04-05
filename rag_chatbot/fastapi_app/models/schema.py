from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    country: str

class ChatResponse(BaseModel):
    answer: str
    