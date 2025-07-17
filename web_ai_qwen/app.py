from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import requests
import os
import uuid
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load Vision-Language model (Qwen2.5-VL)
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # using float16 for efficiency
    device_map="auto",
)
model = torch.compile(model)
processor = AutoProcessor.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_images", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")  # Handle multiple files
    if not files:
        return jsonify({"error": "No images provided"}), 400
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    image_urls = []
    for image_file in files:
        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(upload_folder, filename)
        image_file.save(image_path)
        image_url = f"/static/uploads/{filename}"
        image_urls.append(image_url)
    return jsonify({"image_urls": image_urls})

@app.route("/analyze_images", methods=["POST"])
def analyze_images():
    print(request.json)
    image_urls = request.json.get("image_urls")
    if not image_urls or not isinstance(image_urls, list):
        return jsonify({"error": "No image URLs provided"}), 400
    text_prompt = request.json.get("prompt")
    if not text_prompt:
        return jsonify({"error": "No prompt provided"}), 400
    base_dir = os.path.abspath(os.path.dirname(__file__))
    static_folder = os.path.join(base_dir, 'static')
    uploads_folder = os.path.join(static_folder, 'uploads')
    
    # Prepare messages with all images and the text prompt
    # {'type': 'image', 'image': 'file://E:\\00_ResearchProject\\AI_Implementation\\web_ai_qwen\\static\\uploads\\8acf8d9b-8f8c-458e-b9e2-f44f0ed1b711.jpg'}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.join(uploads_folder, os.path.basename(url))}"}
                for url in image_urls
            ] + [{"type": "text", "text": text_prompt}],
        }
    ]
    print(f'Message: {messages}')
    
    # Format text using chat template processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process vision information (images)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Run inference to generate output text
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f'Response: {output_text[0]}')
    return jsonify({
        "qwen_response": output_text[0]
    })

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)