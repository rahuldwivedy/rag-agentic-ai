from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    result = graph.invoke(
        {"question": req.query, "context": [], "scores": [], "answer": "", "score": 0.0}
    )
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )
