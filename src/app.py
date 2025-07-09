from flask import Flask, request, jsonify, render_template_string
import torch
from model_utils import load_model, process_inputs, decode_outputs
import time
import os
import glob
import PyPDF2

app = Flask(__name__)

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model("Qwen/Qwen2.5-Omni-7B", device_map="auto")

documents = []  # In-memory store for ingested text chunks


def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    return ""


def ingest_file(filepath):
    text = extract_text_from_file(filepath)
    if text:
        documents.append({"filename": os.path.basename(filepath), "text": text})
        return True
    return False


def ingest_folder(folder):
    count = 0
    for ext in ("*.txt", "*.pdf"):
        for filepath in glob.glob(os.path.join(folder, ext)):
            if ingest_file(filepath):
                count += 1
    return count


def retrieve_context(query, top_k=1):
    # Simple keyword search: return the most relevant document chunk
    results = []
    for doc in documents:
        if query.lower() in doc["text"].lower():
            results.append(doc["text"])
    if results:
        return results[:top_k]
    # fallback: return first doc if nothing matches
    return [documents[0]["text"]] if documents else []


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)
    if ingest_file(filepath):
        return jsonify({"status": "success", "filename": file.filename})
    return jsonify({"error": "Failed to ingest file"}), 500


@app.route("/ingest_folder", methods=["POST"])
def ingest_folder_api():
    data = request.json
    folder = data.get("folder")
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Invalid folder path"}), 400
    count = ingest_folder(folder)
    return jsonify({"status": "success", "files_ingested": count})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.json
    conversation = data.get("conversation")
    user_query = ""
    if conversation and len(conversation) > 1:
        user_query = conversation[-1]["content"][0]["text"]
    # Retrieve context from ingested docs
    context_chunks = retrieve_context(user_query) if user_query else []
    if context_chunks:
        # Prepend context to system prompt
        conversation = conversation.copy()
        conversation.insert(
            1,
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"Relevant context from your documents: {context_chunks[0][:1000]}",
                    }
                ],
            },
        )
    if not conversation:
        return jsonify({"error": "No conversation provided"}), 400
    model_device = next(model.parameters()).device
    inputs = process_inputs(conversation, processor)
    inputs = {
        k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()
    }
    start_time = time.time()
    text_ids = model.generate(**inputs, use_audio_in_video=True)
    end_time = time.time()
    # Handle tuple output
    if isinstance(text_ids, tuple):
        token_tensor = text_ids[0]
    else:
        token_tensor = text_ids
    num_tokens = (
        token_tensor.shape[-1] if hasattr(token_tensor, "shape") else len(token_tensor)
    )
    response_time_ms = int((end_time - start_time) * 1000)
    tokens_per_sec = (
        round(num_tokens / (end_time - start_time), 2) if end_time > start_time else 0
    )
    text = decode_outputs(text_ids, processor)
    if isinstance(text, list):
        text = " ".join(text)
    return jsonify(
        {
            "response": text,
            "metrics": {
                "response_time_ms": response_time_ms,
                "tokens_per_sec": tokens_per_sec,
            },
        }
    )


@app.route("/", methods=["GET"])
def index():
    return """
    <html>
    <body>
        <h2>Qwen Chat</h2>
        <form action="/infer" method="post" id="chatForm">
            <textarea name="user_input" rows="4" cols="50" placeholder="Type your message here..."></textarea><br>
            <input type="submit" value="Send">
        </form>
        <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
            <input type="file" name="file" accept=".txt,.pdf" />
            <input type="submit" value="Upload Document">
        </form>
        <div id="spinner" style="display:none;">🌀 Thinking...</div>
        <div id="response"></div>
        <script>
        document.getElementById('chatForm').onsubmit = async function(e) {
            e.preventDefault();
            document.getElementById('spinner').style.display = 'block';
            document.getElementById('response').innerText = '';
            const user_input = document.querySelector('textarea[name="user_input"]').value;
            const conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input}
                    ]
                }
            ];
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({conversation})
            });
            const data = await response.json();
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('response').innerText = data.response +
                (data.metrics ? "\\nResponse time: " + data.metrics.response_time_ms + " ms, Tokens/sec: " + data.metrics.tokens_per_sec : "");
        }
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            const formData = new FormData(document.getElementById('uploadForm'));
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            alert(data.status ? 'Uploaded: ' + data.filename : 'Error: ' + data.error);
        }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
