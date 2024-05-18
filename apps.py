from flask import Flask, render_template, jsonify, request
from src.helper import repo_ingestion, loading_repo_as_documents, chunk_documents, load_embedding_model
from store_index import vectorize
from inferencing import inferencing

app = Flask(__name__)

# Global variable to store vector database
vector_db = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def gitRepo():
    global vector_db
    try:
        if "question" not in request.form:
            return jsonify({"error": "No 'question' field in form data"}), 400

        user_input = request.form["question"]
        repo_path = repo_ingestion(user_input)
        documents = loading_repo_as_documents(repo_path)
        chunks = chunk_documents(documents)
        embedding_model = load_embedding_model()
        vector_db = vectorize(chunks, embedding_model)
        

        response = {"response": "Repository processed successfully and vector database initialized."}
        return jsonify(response)

    except Exception as e:
        print(f"Error processing repository: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/get", methods=["POST"])
def get_response():
    global vector_db
    try:
        if vector_db is None:
            return jsonify({"error": "Vector database is not initialized. Please ingest a repository first."}), 400

        if "msg" not in request.form:
            return jsonify({"error": "No 'msg' field in form data"}), 400

        user_input = request.form["msg"]

        response_text = inferencing(vector_db, user_input)
        return response_text
        

        
        
    except Exception as e:
        print(f"Error during inferencing: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
