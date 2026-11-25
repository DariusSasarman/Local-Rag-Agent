import os.path
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import math

from langchain_core.documents import Document
from tqdm import tqdm

from retrievers.bm25s_retriever import BM25SRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from data.fiqa_download_script import download_data, DATA_DIR
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_STORE_PATH = SCRIPT_DIR / "data"

RERANK_SAMPLE_SIZE = 50
SEED = 42

retrievers = [BM25SRetriever,VectorRetriever,HybridRetriever]

def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain for a single query."""
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def ingest_corpus(retriever, corpus):
    rows = zip(corpus["_id"], corpus["title"], corpus["text"])
    for doc_id, title, text in tqdm(rows, total=len(corpus), desc=type(retriever).__name__):
        doc = Document(
            page_content=f"{title}\n{text}".strip(),
            metadata={"doc_id": str(doc_id)},
        )
        retriever.save_document([doc], f"fiqa/{doc_id}")

def ranked_ids(hits):
    ids = []
    for d in hits:
        i = d.metadata["doc_id"]
        if i not in ids:
            ids.append(i)
    return ids

if __name__ == "__main__":

    if not os.path.exists(DATA_DIR):
        print("Downloading the data...")
        download_data()
    else :
        print("Data exists, skipping the additional download")

    # loading data
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
    sample = queries_with_qrels.sample(n=RERANK_SAMPLE_SIZE, random_state=SEED)
    print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")


    results: dict[str, list[float]] = defaultdict(list)
    for ret_cl in retrievers:
        name = ret_cl.__name__
        store = DATA_STORE_PATH / name
        r = ret_cl(str(store))
        marker = store / ".ingested"

        if not marker.exists():
            ingest_corpus(r,corpus)
            marker.touch()

        search = r.as_retriever(k=10)
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=name):
            hits = search.invoke(row["text"])
            results[name].append(ndcg_at_k(ranked_ids(hits), qrels[str(row["_id"])]))

    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")
