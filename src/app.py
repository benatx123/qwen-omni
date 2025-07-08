from flask import Flask, request, jsonify
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from model_utils import load_model, process_inputs, decode_outputs

app = Flask(__name__)

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model("Qwen/Qwen2.5-Omni-7B", device_map="auto")

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    conversation = data.get("conversation")
    
    if not conversation:
        return jsonify({"error": "No conversation provided"}), 400

    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = process_inputs(
        conversation,
        processor
    )
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    text_ids = model.generate(**inputs, use_audio_in_video=True)
    print("text_ids type:", type(text_ids))
    print("text_ids:", text_ids)
    text = decode_outputs(text_ids, processor)

    # Return as string if list
    if isinstance(text, list):
        text = " ".join(text)

    return jsonify({"response": text})

@app.route('/', methods=['GET'])
def index():
    return '''
    <html>
    <body>
        <h2>Qwen Chat</h2>
        <form action="/infer" method="post" id="chatForm">
            <textarea name="user_input" rows="4" cols="50" placeholder="Type your message here..."></textarea><br>
            <input type="submit" value="Send">
        </form>
        <div id="response"></div>
        <script>
        document.getElementById('chatForm').onsubmit = async function(e) {
            e.preventDefault();
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
            document.getElementById('response').innerText = data.response;
        }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)