import time
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str
    score: float


def _to_text(content):
    """Gemini may return a list of content blocks; join them into one string."""
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
    )


def build_rag_graph():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    vectorstore = PineconeVectorStore(index_name=config.INDEX_NAME, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=1,
    )

    def retrieve(state: AgentState):
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        return {
            "context": [d.page_content for d, _ in results],
            "scores": [float(s) for _, s in results],
        }

    def generate(state: AgentState):
        scores = state["scores"]
        top = max(scores) if scores else 0.0

        # Guardrail: nothing relevant retrieved -> refuse without calling the LLM
        if not state["context"] or top < config.MIN_RELEVANCE:
            return {"answer": config.REFUSAL, "score": round(top, 3)}

        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question using ONLY the context below.
Do not use outside knowledge. If the context does not contain enough information,
reply exactly: "{config.REFUSAL}"

Context:
{context_str}

Question: {state['question']}
Answer:"""

        for attempt in range(2):  # one retry on temporary Gemini errors
            try:
                resp = llm.invoke(prompt)
                break
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(5)
        answer = _to_text(resp.content).strip()

        if config.REFUSAL.lower() in answer.lower():
            return {"answer": config.REFUSAL, "score": 0.0}

        confidence = round(sum(scores) / len(scores), 3)  # mean similarity of retrieved chunks
        return {"answer": answer, "score": confidence}

    wf = StateGraph(AgentState)
    wf.add_node("retrieve", retrieve)
    wf.add_node("generate", generate)
    wf.add_edge(START, "retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", END)
    return wf.compile()