import time
import requests

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # validation: should be refused
]

lines = []
for q in QUERIES:
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/chat", json={"query": q}, timeout=180
        )
    except requests.exceptions.RequestException as e:
        out = f"\nQ: {q}\nERROR: {e}"
        print(out)
        lines.append(out)
        continue

    if resp.status_code != 200:
        out = f"\nQ: {q}\nERROR {resp.status_code}: {resp.text[:300]}"
    else:
        r = resp.json()
        out = (
            f"\nQ: {q}\nA: {r['answer']}\nConfidence: {r['confidence_score']}"
            f"\nChunks retrieved: {len(r['retrieved_chunks'])}"
        )
    print(out)
    lines.append(out)
    time.sleep(10)  # stay within free-tier rate limits

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved results to test_results.txt")