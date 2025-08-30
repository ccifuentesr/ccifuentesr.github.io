import os
from flask import Flask, request, jsonify, send_from_directory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

load_dotenv()  # loads .env

api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)

# --- Directorios ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
FRONTEND_DIR = os.path.join(BASE_DIR, "..")

# --- Carga FAISS ---
db = FAISS.load_local(
    FAISS_DIR,
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 5})  # Return more documents

# --- QA chain ---
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant specialised in astrophysics.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

"""
)


qa = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, openai_api_key=api_key),
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt}
)

# --- Endpoints ---
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "science.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"[DEBUG] Question received: {question}")
        
        # Debug: check what documents are retrieved
        docs = retriever.invoke(question)
        print(f"[DEBUG] Retrieved {len(docs)} documents")
        if docs:
            print(f"[DEBUG] First doc preview: {docs[0].page_content[:200]}...")
        
        answer = qa.invoke({"query": question})["result"]
        print(f"[DEBUG] Answer generated successfully")
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"[ERROR] In ask endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)