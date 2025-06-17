# import json
# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# import requests
# import os
# import uuid
# import torch
# from transformers import BlipProcessor, BlipForQuestionAnswering

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

@app.route("/upload_image", methods=["POST"])
def upload_image():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(upload_folder, filename)
    image_file.save(image_path)

    # Buat URL valid
    image_url = f"/static/uploads/{filename}"
    return jsonify({"image_url": image_url})

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    print(request.json)
    image_url = request.json.get("image_url")
    print(image_url)
    if not image_url:
        print('Here 1')
        return jsonify({"error": "No image URL provided"}), 400

    text_prompt = request.json.get("prompt")
    if not text_prompt:
        print('Here 2')
        return jsonify({"error": "No prompt provided"}), 400

    # Pastikan path absolut ke folder 'static/uploads/'
    base_dir = os.path.abspath(os.path.dirname(__file__))
    static_folder = os.path.join(base_dir, 'static')
    uploads_folder = os.path.join(static_folder, 'uploads')

    # Ubah URL menjadi path lokal
    local_image_path = os.path.join(uploads_folder, os.path.basename(image_url))
    if not os.path.exists(local_image_path):
        return jsonify({"error": f"File does not exist: {local_image_path}"}), 400

    try:
        image = Image.open(local_image_path).convert('RGB')
    except FileNotFoundError:
        return jsonify({"error": f"Image file not found at {local_image_path}"}), 400

    # Persiapkan pesan input dengan gambar dan teks pertanyaan.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{local_image_path}"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Format teks menggunakan template chat processor.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Proses informasi visi (gambar).
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Jalankan inferensi untuk menghasilkan teks output.
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return jsonify({
        "qwen_response": output_text[0]
    })

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)