from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# 加载模型和处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to("cuda")

# 加载图像
image = Image.open("/root/autodl-tmp/datasets/coco/val2014/COCO_val2014_000000522418.jpg")

# 生成描述
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_length=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"生成的描述: {generated_text}")
