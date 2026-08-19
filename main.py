from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="Bloom Postpartum Companion API")

# This allows your React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",         # local dev
        "https://your-app.lovable.app",  # replace with your actual Lovable URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Bloom API"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))