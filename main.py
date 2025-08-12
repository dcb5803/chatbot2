# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

class QARequest(BaseModel):
    context: str
    question: str

@app.post("/answer")
async def get_answer(data: QARequest):
    if not data.context.strip() or not data.question.strip():
        return {"answer": "Please provide both context and a question."}
    result = qa_pipeline(question=data.question, context=data.context)
    return {"answer": result["answer"]}

# Optional for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
