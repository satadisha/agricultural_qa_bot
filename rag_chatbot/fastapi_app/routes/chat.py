from fastapi import APIRouter
from fastapi_app.models.schema import ChatRequest, ChatResponse
from fastapi_app.services.rag_pipeline import get_chat_response

chat_router = APIRouter()

@chat_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = get_chat_response(request.query, request.country)
    return ChatResponse(answer=response)

