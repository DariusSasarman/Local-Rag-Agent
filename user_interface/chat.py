import pandas as pd

from retrievers.hybrid_retriever import HybridRetriever
from models.language_model import LocalQwen
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gc, torch
import gradio as gr
import os

COMMON_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runtime")
DB_PATH = os.path.join(COMMON_PATH,"data")

condense_prompt = PromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite the follow-up
as a standalone question. Return only the question.

History:
{history}

Follow-up: {question}

Standalone question:""")

answer_prompt = PromptTemplate.from_template("""
Use the context to answer the question. If you don't know, say so.
Use three sentences maximum.

Context: {context}

Chat history:
{history}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history, max_turns=4):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-max_turns * 2:])

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def condense_history(llm_used, query, history_inp):
    condense = condense_prompt | llm_used | StrOutputParser()
    llm_used.load()
    condensed_history = condense.invoke({"question": query, "history": history_inp}).strip()
    llm_used.unload()
    return condensed_history

def generate_answer(llm_used, docs_inp, history_inp, query : str) -> str:
    answer_chain = answer_prompt | llm_used | StrOutputParser()

    llm_used.load()
    ret_answer = answer_chain.invoke({
        "context": format_docs(docs_inp),
        "history": history_inp,
        "question": query,
    }).strip()
    llm_used.unload()

    return ret_answer

def ingest_documents(files, progress=gr.Progress()):
    if not files:
        yield "No files selected.", pd.DataFrame()
        return

    log, rows = "", []
    hybrid_retriever = HybridRetriever(DB_PATH)
    try:
        for f in progress.tqdm(files, desc="Ingesting"):
            try:
                hybrid_retriever.ingest_document(f)
                rows.append({"file": f, "status": "ok"})
                log += f"OK {f}\n"
            except Exception as e:
                rows.append({"file": f, "status": str(e)})
                log += f"FAIL {f}: {e}\n"
            yield log, pd.DataFrame(rows)
    finally:
        del hybrid_retriever
        free_gpu()

def retrieve_documents(context_query : str = ""):
    hybrid_retriever = HybridRetriever(DB_PATH)
    docs_ret = hybrid_retriever.as_retriever().invoke(context_query)
    del hybrid_retriever
    free_gpu()
    return docs_ret

def respond(message, history):
    llm = LocalQwen()
    hist = format_history(history)
    standalone = condense_history(llm, message, hist) if history else message
    standalone = standalone or message
    docs = retrieve_documents(standalone)
    return generate_answer(llm, docs, hist, message)

if __name__ == "__main__":
    os.makedirs(COMMON_PATH,exist_ok=True)
    os.chdir(COMMON_PATH)
    with gr.Blocks() as demo:
        with gr.Tab("Chat"):
            gr.ChatInterface(fn=respond)
        with gr.Tab("Document Database"):
            files = gr.File(file_count="multiple", label="Documents", type="filepath")
            btn = gr.Button("Ingest", variant="primary")
            log = gr.Textbox(label="Log", lines=8)
            table = gr.Dataframe(label="Results")
            btn.click(ingest_documents, inputs=files, outputs=[log, table])

    demo.launch()