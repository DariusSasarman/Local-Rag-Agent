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
    return "\n".join(f"{role}: {msg}" for role, msg in history[-max_turns * 2:])

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

def retrieve_documents(hybrid_db_location : str = "", context_query : str = ""):
    hybrid_retriever = HybridRetriever()
    docs_ret = hybrid_retriever.as_retriever().invoke(context_query)
    del hybrid_retriever
    free_gpu()
    return docs_ret

def respond(message, history):
    llm = LocalQwen()
    hist = format_history(history)
    standalone = condense_history(llm, message, hist) if history else message
    standalone = standalone or message
    docs = retrieve_documents(DB_PATH,standalone)
    return generate_answer(llm, docs, hist, message)

if __name__ == "__main__":
    os.makedirs(COMMON_PATH,exist_ok=True)
    os.chdir(COMMON_PATH)
    gr.ChatInterface(fn=respond).launch(theme="ocean")