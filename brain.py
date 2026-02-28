from fastapi import APIRouter
import ollama

router = APIRouter()

@router.post("/query")
async def query_brain(payload: dict):
    response = ollama.chat(
        model="paul-brain",
        messages=[{"role": "user", "content": payload.get("intent", "")}]
    )
    return {
        "response": response["message"]["content"],
        "model": "paul-brain",
        "entity": "Paulk",
        "status": "ok"
    }
