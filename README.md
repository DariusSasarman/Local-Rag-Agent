# Local-Rag-Agent

I'm studying Generative AI technologies and this repo serves as my practical exercise workspace.

This is the current architecture
```mermaid
flowchart LR
    subgraph di[Data Ingestion]
        PFD["process_folder()"] --> File["ingest_document()"] 
        File --> Save["save_document()"]
    end
    di --> rt
    subgraph rt[Runtime execution loop]
        st[Start] --> chk{History not null?}
        chk -->|yes| condense["condense.invoke()"]
        chk -->|no| retrieve["hybrid_db.as_retriever().invoke()"]
        condense --> retrieve
        retrieve --> vec_ret["vec_ret.as_retriever.invoke()"]
        retrieve --> key_ret["key_ret.as_retriever.invoke()"]
        vec_ret --> r_kkf["_reciprocal_rank_fusion()"]
        key_ret --> r_kkf["_reciprocal_rank_fusion()"]
        r_kkf --> crs_enc["reranker.rerank()"]
        crs_enc --> ans["answer_chain.invoke()"]
        ans --> prin["print('Answer...')"]
        prin --> st
    end
```