# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="mps",
)
processor = AutoProcessor.from_pretrained(model_id)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
# image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
# image_url = "https://cdn2.techbang.com/system/excerpt_images/22233/original/2773841ffa307c4e2bcf80e6b8be9583.jpg"# 鑰匙
# image_url = "https://img.ltn.com.tw/Upload/news/600/2020/01/21/3046890_1_1.jpg"# 貓頭鷹
image_url = "https://fishdb.sinica.edu.tw/chi/xray/large/Xray_ASIZP0056171.jpg"# 石斑魚
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist.You are native Chinese."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray with Chinese"},
            {"type": "image", "image": image}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)