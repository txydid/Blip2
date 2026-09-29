import re

with open('lavis/models/blip2_models/blip2_qformer.py', 'r') as f:
    content = f.read()

# 在 BertSelfAttention 的 __init__ 开头添加类型转换
pattern = r'(class BertSelfAttention.*?def __init__\(self, config.*?\):)(.*?)(self\.num_attention_heads)'

def replacement(match):
    prefix = match.group(1)
    middle = match.group(2)
    rest = match.group(3)
    
    # 添加类型转换
    type_fix = '\n        # Ensure encoder_width is int\n        if hasattr(config, "encoder_width") and isinstance(config.encoder_width, str):\n            config.encoder_width = int(config.encoder_width)\n        '
    
    return prefix + middle + type_fix + rest

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('lavis/models/blip2_models/blip2_qformer.py', 'w') as f:
    f.write(content)

print("✅ 已修复 encoder_width 类型问题")
