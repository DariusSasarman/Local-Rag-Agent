
from vector_database import Vector_db
from language_model import LocalQwen
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import gc, torch

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

if __name__ == "__main__":
    llm = LocalQwen()
    condense = condense_prompt | llm | StrOutputParser()
    answer_chain = answer_prompt | llm | StrOutputParser()
    history = []

    while True:
        q = input("\nYour Question: ")
        if q.lower() in ["exit", "quit", "q"]:
            break
        hist = format_history(history)

        # 1. LLM: condense follow-up
        if history:
            llm.load()
            standalone = condense.invoke({"question": q, "history": hist}).strip()
            llm.unload()
        else:
            standalone = q
        if not standalone:
            standalone = q

        # 2. Encoder: retrieve
        db = Vector_db()       # loads the encoder
        docs = db.as_retriever().invoke(standalone)
        del db
        free_gpu()

        # 3. LLM: answer
        llm.load()
        answer = answer_chain.invoke({
            "context": format_docs(docs),
            "history": hist,
            "question": q,
        }).strip()
        llm.unload()

        history += [("User", q), ("Assistant", answer)]
        print(f"Answer: {answer}")