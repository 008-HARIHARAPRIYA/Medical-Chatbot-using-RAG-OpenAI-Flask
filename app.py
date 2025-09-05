from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings, load_documents  # Ensure you have a function to load documents
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os
from src.prompt import system_prompt
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set environment variables for libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Load embeddings and documents
embeddings = download_embeddings()
documents = load_documents()  # You need to define this in src.helper
index_name = "medical-bot"

# Create Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create local Hugging Face pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# Wrap pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg:
        return jsonify({"error": "No input provided"}), 400

    print("User Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return jsonify({"answer": response["answer"]})

@app.route("/")
def index():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
