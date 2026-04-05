from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.middleware.cors import CORSMiddleware
from fastapi_app.routes.chat import chat_router

import asyncio
import uuid
from fastapi_app.services.rag_pipeline import get_chat_response

 


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.include_router(chat_router)


# Store tasks and their results
# { task_id: {"status": "waiting" | "processing" | "completed", "answer": str, "question": str} }
tasks = {}
task_queue = asyncio.Queue()

# HTML + CSS setup
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")
templates = Jinja2Templates(directory="fastapi_app/templates")

# Background Worker
async def worker():
    while True:
        task_id = await task_queue.get()
        task_data = tasks[task_id]
        task_data["status"] = "processing"
        
        try:
            # We use to_thread so the heavy LLM work doesn't freeze the whole server
            answer = await asyncio.to_thread(
                get_chat_response, 
                query=task_data["question"], 
                country=task_data["country"]
            )
            tasks[task_id].update({"status": "completed", "answer": answer})
        except Exception as e:
            tasks[task_id].update({"status": "failed", "answer": f"Error: {str(e)}"})
        
        task_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    print("get request: connecting")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(question: str = Form(...), country: str = Form(...)):
    print("post request: /ask")
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "waiting",
        "question": question,
        "country": country,
        "answer": None
    }
    await task_queue.put(task_id)
    return {"task_id": task_id}
    #return templates.TemplateResponse("index.html", {"task_id": task_id,"request": Request})

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    print("get request: /status")
    if task_id not in tasks:
        return HTMLResponse({"error": "Task not found"}, status_code=404)

    # Calculate position in queue
    queue_list = list(task_queue._queue)
    position = queue_list.index(task_id) + 2 if task_id in queue_list else 0

    return {
        "status": tasks[task_id]["status"],
        "position": position,
        "answer": tasks[task_id]["answer"],
        "question": tasks[task_id]["question"]
    }


# uvicorn fastapi_app.main:app --reload
