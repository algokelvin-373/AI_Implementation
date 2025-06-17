from PIL import Image
import requests
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


# Download and process the demo image
demo_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(demo_url, stream=True).raw).reduce(2).convert('RGB')
image.save('demo.jpeg')

# Load the model with torch.float16 and compile it for optimization.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,  # using float16 here for efficiency
    device_map="auto",
)
model = torch.compile(model)

# Load the processor that handles both image and text input.
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Prepare the message input with the demo image and text query.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file://demo3.jpg"},  # Use the prepared demo image
            {"type": "text", "text": "what occasion can be used"},
        ],
    }
]

# Format the text using the processor's chat template.
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# Process vision (image) information.
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Run inference to generate output text.
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Generated Description:", output_text)