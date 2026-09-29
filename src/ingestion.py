import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config


def ensure_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if config.INDEX_NAME not in existing:
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        try:  # start clean so reruns never create duplicates
            pc.Index(config.INDEX_NAME).delete(delete_all=True)
        except Exception:
            pass


def run_ingestion():
    ensure_index()
    docs = PyPDFLoader(config.PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    store = PineconeVectorStore(index_name=config.INDEX_NAME, embedding=embeddings)

    for i in range(0, len(chunks), config.BATCH_SIZE):
        batch = chunks[i : i + config.BATCH_SIZE]
        ids = [f"chunk-{i + j}" for j in range(len(batch))]
        for attempt in range(3):  # retry on rate-limit errors
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as e:
                print(f"Retrying batch after error: {str(e)[:80]}")
                time.sleep(30)
        else:
            raise RuntimeError("Batch failed after 3 retries")
        print(f"Upserted {min(i + config.BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
        time.sleep(config.BATCH_PAUSE)

    print(f"Done: {len(docs)} pages -> {len(chunks)} chunks.")


if __name__ == "__main__":
    run_ingestion()