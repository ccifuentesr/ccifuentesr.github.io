import os
from flask import Flask, request, jsonify, send_from_directory

# Try to import langchain/FAISS-related modules; allow them to be absent so the
# server can still run for lightweight endpoints like /vizier.
HAS_LANGCHAIN = False
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except Exception as e:
    print(f"[WARN] langchain/FAISS imports failed: {e}")

from dotenv import load_dotenv
from flask_cors import CORS
import requests
import csv
from io import StringIO


load_dotenv() # loads .env
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# --- CORS Configuration ---
# Allow requests from your GitHub Pages domain
CORS(app,
origins=["https://ccifuentesr.github.io", "http://localhost:*"],
supports_credentials=True,
methods=["GET", "POST", "OPTIONS"],
allow_headers=["Content-Type"])

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
FRONTEND_DIR = os.path.join(BASE_DIR, "..")

retriever = None
qa = None
custom_prompt = None

if HAS_LANGCHAIN:
    try:
        # --- Load FAISS ---
        db = FAISS.load_local(
            FAISS_DIR,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 5}) # Return more documents

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

Instructions:
- You must refer to the context as a thesis.
- Use only information from the thesis.
- If the answer is in the thesis, give it clearly and concisely.
- If the thesis lacks the information, state this and then provide a grounded answer.
- Use HTML for maths (e.g. R<sub>⊙</sub>).
- Do not use LaTeX.
- Keep a formal, scientific tone.
"""
        )

        qa = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(model="gpt-5-mini", max_tokens=800, openai_api_key=api_key),
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": custom_prompt}
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize FAISS/QA chain: {e}")
        HAS_LANGCHAIN = False

# --- Endpoints ---
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "science.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        data = request.json
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"[DEBUG] Question received: {question}")
        
        if not HAS_LANGCHAIN or retriever is None or qa is None:
            return jsonify({"error": "QA functionality is not available on this server (missing langchain/FAISS)."}), 503

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


@app.route("/vizier", methods=["GET"])
def vizier_proxy():
    """Proxy endpoint to query VizieR and return a compact JSON summary.
    Query params:
      - source: VizieR source code (e.g. J/A+A/693/A228)
      - term: target name (e.g. GJ 2)
    Returns: JSON with first match fields (name, spt, class, planets) when available.
    """
    source = request.args.get('source', '').strip()
    term = request.args.get('term', '').strip()

    if not term or not source:
        return jsonify({"error": "Missing 'source' or 'term' query parameter"}), 400

    # This endpoint was removed to avoid backend dependency on VizieR.
    return jsonify({"error": "Vizier proxy removed on server. Use client-side search."}), 410

if __name__ == "__main__":
    app.run(debug=True, port=5001)