from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

# # Access the variables
# literal_api_key = os.getenv("LITERAL_API_KEY")
# chainlit_auth_secret = os.getenv("CHAINLIT_AUTH_SECRET")

# # You can now use these variables in your code
# print("Literal API Key:", literal_api_key)
# print("Chainlit Auth Secret:", chainlit_auth_secret)

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """You are a helpful assistant. Use the following conversation history and context to provide a concise answer.

Conversation History: {chat_history}

Context: {context}

Question: {question}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['chat_history', 'context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={
            "max_new_tokens": 2048,
            "context_length": 4096,
            "temperature": 0.5
        }
    )
    
    return llm

def initialize_memory():
    # Initialize memory with a window of the last 2 exchanges
    return ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", input_key="question", output_key="answer", return_messages=True
    )


# Function to compute similarity between new question and previous conversation
def compute_similarity(question, previous_question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Use embed_query for embedding individual questions
    question_embedding = embeddings.embed_query(question)
    prev_embedding = embeddings.embed_query(previous_question)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([question_embedding], [prev_embedding])[0][0]
    return similarity


# Function to check if a topic change has occurred
def is_topic_change(new_question, chat_history):
    last_question = chat_history[-1].content if chat_history else ""
    similarity_score = compute_similarity(new_question, last_question)
    
    return similarity_score < 0.15  # Adjust this threshold as needed

# Function to reset memory based on topic change
def reset_memory(new_question, memory):
    if is_topic_change(new_question, memory.chat_memory.messages):
        memory.clear()

def retrieval_qa_chain(llm, db, memory):
    prompt = set_custom_prompt()
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        memory=memory,  # Memory for conversation history
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},  # Pass the custom prompt here
        verbose=True  # Optional for debugging
    )
    return qa_chain

def qa_bot(memory):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()

    # Pass memory when setting up the retrieval chain
    return retrieval_qa_chain(llm, db, memory)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}").send()
    
    # Initialize memory and store it in the session
    memory = initialize_memory()
    cl.user_session.set("memory", memory)

    # Set up the chain
    chain = qa_bot(memory)
    cl.user_session.set("chain", chain)

    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content = "Hi, Welcome to the Dr. Vigor. How can I help you?"
    await msg.update()


@cl.on_chat_resume
async def resume():
    memory = cl.user_session.get("memory")  # Retrieve memory stored in the session
    chain = cl.user_session.get("chain")    # Retrieve chain stored in the session

    # Check if the memory or chain is not set (in case of a new session)
    if not memory:
        memory = initialize_memory()  # Initialize memory if missing
        cl.user_session.set("memory", memory)  # Save memory to the session

    if not chain:
        chain = qa_bot(memory)  # Recreate chain if missing
        cl.user_session.set("chain", chain)  # Save chain to the session


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # Retrieve the chain from the session
    memory = cl.user_session.get("memory")  # Retrieve the memory from the session

    # Check if the user wants to reset the memory manually
    if message.content.strip().lower() == "/reset":
        chain.memory.clear()  # Clear the memory
        await cl.Message(content="Memory reset! You can now start a new conversation.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # Retrieve chat history from the memory (or leave it empty if it's the first message)
    chat_history = memory.chat_memory.messages if memory else []

    # Reset memory if a topic change is detected
    reset_memory(message.content, memory)

    # Prepare the inputs expected by ConversationalRetrievalChain
    inputs = {
        "question": message.content,  # Pass the message content as 'question'
        "chat_history": chat_history  # Pass the current conversation history
    }

    # Now pass the extracted query to the chain asynchronously
    res = await chain.acall(inputs, callbacks=[cb])

    answer = res.get("answer", "No answer found")
    sources = res["source_documents"]
    
    if not sources:
        answer += f"\nNo Sources Found."

    await cl.Message(content=answer).send()
