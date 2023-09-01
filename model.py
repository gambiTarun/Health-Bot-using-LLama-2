from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DATABASE_PATH = 'vectorstore/faiss_db'

custom_qa_template = """Please consider the information below to respond to the user's inquiry.
If the answer is not known, state it clearly. Do not try to guess.

Context: {context}
Question: {question}

Only return helpful answer and nothing else.
Helpful answer:
"""

def custom_qa_prompt():
    qa_prompt = PromptTemplate(
        template=custom_qa_template,
        input_variables=['context', 'question']
    )
    return qa_prompt

def configure_qa_chain(model, qa_template, faiss_db):
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_config='config_name',
        retriever=faiss_db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_template}
    )
    return chain

def initialize_model():
    language_model = CTransformers(
        model="/Users/tarunbirgambhir/Documents/Experiments/Medical Chatbot using Llama 2/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return language_model

def setup_qa_bot():
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DATABASE_PATH, embed_model)
    language_model = initialize_model()
    prompt = custom_qa_prompt()
    configured_chain = configure_qa_chain(language_model, prompt, db)

    return configured_chain

def query_response(query_text):
    query_chain = setup_qa_bot()
    processed_response = query_chain({'query': query_text})
    return processed_response

# ChainLit code

@cl.on_chat_start
async def initiate():
    active_chain = setup_qa_bot()
    initial_message = cl.Message(content="Initializing the bot...")
    await initial_message.send()
    initial_message.content = "Hello, welcome to the HealthBot. What would you like to ask?"
    await initial_message.update()

    cl.user_session.set("active_chain", active_chain)

@cl.on_message
async def process_query(user_message):
    active_chain = cl.user_session.get("active_chain")
    async_cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    async_cb.answer_reached = True
    results = await active_chain.acall(user_message, callbacks=[async_cb])
    result_answer = results["result"]
    result_docs = results["source_documents"]

    if result_docs:
        result_answer += f"\nDocument Sources:" + str(result_docs)
    else:
        result_answer += "\nNo relevant documents found."

    await cl.Message(content=result_answer).send()
