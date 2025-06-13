import json

from flask import Flask, request, jsonify, render_template
from PIL import Image
import requests
import os
import uuid
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

app = Flask(__name__, template_folder='templates')

# Load BLIP model dan processor (hanya sekali saat server start)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def describe_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    question = "What is in the image?"
    inputs = processor(raw_image, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def get_full_response(prompt):
    full_response = ""
    try:
        with requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5vl:7b",
            "prompt": prompt,
            "stream": True  # Aktifkan streaming
        }, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except Exception as e:
                        print(f"Error parsing response: {e}")
                        continue
    except Exception as e:
        print(f"Error sending request to Ollama: {e}")
        return None
    return full_response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    # Simpan gambar
    if not os.path.exists('images'):
        os.makedirs('images')
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join('images', filename)
    image_file.save(image_path)

    text_prompt = request.form.get("prompt")
    if not text_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Deskripsikan gambar
    image_description = describe_image(image_path)
    print(f"Image Description: {image_description}")  # Logging

    # Buat prompt lengkap
    full_prompt = f"The image contains: {image_description}. Based on this description, provide a detailed analysis of the scene, including possible interactions between the dog and cat, their positions, and any other notable elements in the image."
    print(f"Full Prompt: {full_prompt}")  # Logging

    # Kirim ke Qwen2.5VL
    qwen_response = get_full_response(full_prompt)
    if qwen_response is None:
        return jsonify({"error": "Failed to get response from Qwen2.5VL"}), 500

    return jsonify({
        "image_description": image_description,
        "qwen_response": qwen_response
    })

if __name__ == "__main__":
    app.run(debug=True)