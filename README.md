# RAG Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation chatbot that answers **only** from the Agentic AI eBook. Built with Python, LangGraph, Pinecone, Google Gemini and FastAPI.

> Note: the reference suggested OpenAI; I used Google Gemini (`gemini-embedding-001` for embeddings, `gemini-3.1-flash-lite` for generation) because it has a free API tier. The pipeline is model-agnostic.

## Architecture

```
Ingestion (run once):  PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000/200)
                       -> Gemini gemini-embedding-001 (3072-d) -> Pinecone (cosine)

Query (LangGraph):     POST /chat -> START -> retrieve -> generate -> END -> JSON
```

- `src/config.py`: env vars and constants (chunk size, top-k, threshold, models).
- `src/ingestion.py`: creates the Pinecone index if missing, loads, chunks, embeds and upserts the PDF.
- `src/graph.py`: LangGraph `StateGraph` with `AgentState` (question, context, scores, answer, score).
  - `retrieve`: top-k similarity search returning chunks and cosine scores.
  - `generate`: refuses if the best score is below `MIN_RELEVANCE`; otherwise calls Gemini (temperature 0) with a strict "context only" prompt.
- `app.py`: FastAPI `/chat` endpoint.

## Grounding and confidence score

- Strict prompt: answer only from context, otherwise return the fixed refusal string.
- Pre-LLM guardrail: if the top similarity is below `MIN_RELEVANCE` (0.50), the LLM is not called.
- Confidence score = mean cosine similarity of the retrieved chunks (0 to 1). It is 0.0 when the model refuses.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your Gemini and Pinecone keys
```

Download the eBook (https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view) and save it as `data/Ebook-Agentic-AI.pdf`.

## Run

```bash
python -m src.ingestion          # ingest once
uvicorn app:app --reload         # start API on :8000
python tests_sample_queries.py   # in a second terminal
```

Example request:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Response: `{"answer": "...", "retrieved_chunks": ["..."], "confidence_score": 0.0}`

## Test results

## Test results

Run with `python tests_sample_queries.py` against the running API (5 queries, including one out-of-scope validation query).

```text
Q: What is Agentic AI according to the eBook?
A: According to the eBook, Agentic AI is a system that goes beyond other AI by acting autonomously to achieve goals. It is impact-focused, proactive, and adaptive, characterized by its ability to learn continuously, focus on goals, act independently, anticipate needs, adapt to change, and align with goals.
Confidence: 0.785
Chunks retrieved: 4

Q: How do AI agents differ from traditional automation systems?
A: AI agents differ from traditional automation systems in that they are goal-driven systems capable of performing actions autonomously in a dynamic environment, whereas traditional automation systems (such as RPA) excel at repetitive, rule-based tasks with structured data by executing predefined rules and logic.
Confidence: 0.748
Chunks retrieved: 4

Q: What are the core components of an Agentic Architecture?
A: The core components of an Agentic Architecture are: Perception, Reasoning, Planning, Learning, and Execution.
Confidence: 0.778
Chunks retrieved: 4

Q: What role does memory play in Agentic AI workflows?
A: Memory allows an agent to store long-term information about past interactions and successful methods for completing tasks, which helps reduce the computing required for new tasks by referencing relevant past plans and actions. It also serves as a place to store human demonstrations, which can expedite progress without requiring as rigorous a planning and reasoning loop. Additionally, short-term memory, consisting of current context from prompts and available information, allows the agent to address immediate concerns effectively.
Confidence: 0.758
Chunks retrieved: 4

Q: Who won the 2022 FIFA World Cup?
A: I cannot answer based on the provided document.
Confidence: 0.0
Chunks retrieved: 4
```

**Observations**
- In-document questions return grounded answers with confidence between 0.75 and 0.79 (mean cosine similarity of the 4 retrieved chunks).
- The out-of-scope FIFA question is refused with confidence 0.0, showing the strict-grounding prompt works.
