from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests

# 使用官方权重（flan-t5-xl，符合你的环境）
model_id = "Salesforce/blip2-flan-t5-xl-stage2"

processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda")

# 一个小的图文示例
url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "Describe this image."

inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)

outputs = model(**inputs)
generated = processor.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)

print("Output:", generated)

