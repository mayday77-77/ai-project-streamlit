# vectordb.py
import chromadb
from pathlib import Path
from helper_functions.llm import openai_client
from helper_functions.rag import hybrid_chunk_pdf

# global vars
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "policies"
EMBED_MODEL = "text-embedding-3-small"
DATA_DIR = "data"
_CLIENT = None
_COLLECTION = None


# embeddings
def embed_texts(_texts, _model=EMBED_MODEL, _batch_size=64):
    vectors = []
    client = openai_client()
    for index in range(0, len(_texts), _batch_size):
        batch = _texts[index : index + _batch_size]

        resp = client.embeddings.create(
            model=_model,
            input=batch
        )

        for item in resp.data:
            vectors.append(item.embedding)

    return vectors


# create or get collection
def get_or_create_collection(_client, _collection_name):
    collection = _client.get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# build vectordb from chunks
def build_chroma_from_chunks(_all_chunks, _persist_dir, _collection_name, _force_rebuild=False):
    client = chromadb.PersistentClient(path=_persist_dir)

    if _force_rebuild:
        try:
            client.delete_collection(_collection_name)
            print(f"Deleted existing collection '{_collection_name}'")
        except Exception as e:
            print(f"Delete skipped for collection '{_collection_name}': {e}")

    collection = get_or_create_collection(client, _collection_name)

    existing = collection.count()
    if existing > 0 and not _force_rebuild:
        print(f"Collection already has {existing} records — skipping.")
        return collection

    if existing > 0 and _force_rebuild:
        collection.delete(where={})
        print("Cleared old vectors")

    if not _all_chunks:
        print("No chunks provided.")
        return collection

    ids = [f"{chunk.policy_id}_{chunk.chunk_index}" for chunk in _all_chunks]
    documents = [chunk.text for chunk in _all_chunks]
    metadata = [chunk.to_dict() for chunk in _all_chunks]

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed_texts(documents)

    print("Adding to ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadata,
        documents=documents
    )

    print(f"ChromaDB build complete and stored at: {_persist_dir}/")
    return collection


# loads existing vectordb collection
def load_existing_collection(_persist_dir=PERSIST_DIR, _collection_name=COLLECTION_NAME):
    global _CLIENT, _COLLECTION

    # Create client only once
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=_persist_dir)

    # Load collection only once
    if _COLLECTION is None:
        _COLLECTION = _CLIENT.get_collection(_collection_name)
        print(f"Loaded collection '{_collection_name}', total records: {_COLLECTION.count()}")

    return _COLLECTION


def list_policy_ids():
    """ Return a sorted list of distinct policy_id values stored in vectordb """

    client = chromadb.PersistentClient(path="vectordb")
    col = get_or_create_collection(client, 'policies')

    # Fetch *all* metadata entries from Chroma
    # We query empty text with a large limit to retrieve all documents.
    results = col.get(include=["metadatas"], limit=50000)

    policy_ids = set()
    for meta in results.get("metadatas", []):
        pid = meta.get("policy_id")
        if pid:
            policy_ids.add(pid)

    return sorted(policy_ids)


def create_vector_db():
    pdf_dir = Path(DATA_DIR)
    # all PDF files in data folder
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    all_chunks = []

    for pdf in pdfs:
        print(f"\nProcessing {pdf} ...")
        chunks = hybrid_chunk_pdf(str(pdf))
        all_chunks.extend(chunks)
        print(f"Total chunks created: {len(chunks)}")
        # quick sanity: show if merged definitions chunk exists
        #merged = [c for c in chunks if c.section == "definitions_merged"]
        #if merged:
        #    print(f"  → Definitions merged chunk pages {merged[0].page_start}-{merged[0].page_end}")

    build_chroma_from_chunks(
        _all_chunks=all_chunks,
        _persist_dir="vectordb",
        _collection_name="policies",
        #_force_rebuild=True,  # set True to regenerate DB
    )