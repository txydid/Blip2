from transformers import Blip2Processor, Blip2ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from PIL import Image
import json

# 自定义数据集
class COCOCaptionDataset(Dataset):
    def __init__(self, annotation_file, image_dir, processor):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = f"{self.image_dir}/{ann['image']}"
        image = Image.open(image_path).convert('RGB')
        
        # 处理图像和文本
        encoding = self.processor(
            images=image,
            text=ann['caption'],
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        
        # 移除 batch 维度
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        
        return encoding

# 加载模型
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# 创建数据集
train_dataset = COCOCaptionDataset(
    annotation_file="/root/autodl-tmp/datasets/coco/annotations/coco_karpathy_train.json",
    image_dir="/root/autodl-tmp/datasets/coco",
    processor=processor
)

# 训练参数
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/blip2-coco-finetune",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=4,
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
