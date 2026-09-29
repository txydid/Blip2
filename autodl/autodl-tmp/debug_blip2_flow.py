# debug_blip2_flow.py -- robust dtype management for BLIP-2 flow test
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# sample image path (change if you want another image)
sample_img_path = "/root/autodl-tmp/datasets/coco/val2014/COCO_val2014_000000391895.jpg"
use_real_image = os.path.exists(sample_img_path)

print("Loading model & preprocessors ...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device,
)
model.eval()
print("Model loaded.")

# Prepare image tensor
if use_real_image:
    raw_image = Image.open(sample_img_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print("Using real image:", sample_img_path)
else:
    image = torch.randn(1, 3, 224, 224).to(device)
    print("Using random image tensor")

print("Initial image dtype:", image.dtype)

# --- Ensure consistent dtype within visual encoder: make visual encoder & ln_vision half
print("\n--- Casting visual encoder (and ln_vision) to half() to match weights that are half ---")
try:
    model.visual_encoder.half()
    model.ln_vision.half()
    print("visual_encoder and ln_vision cast to half()")
except Exception as e:
    print("Could not cast visual encoder to half():", e)

# Print a few parameter dtypes for inspection (first layers)
def sample_param_dtypes(module, name, max_print=8):
    printed = 0
    for n, p in module.named_parameters():
        print(f"{name}.{n}: {p.dtype}")
        printed += 1
        if printed >= max_print:
            break

print("\n-- Visual encoder param dtypes (sample):")
try:
    sample_param_dtypes(model.visual_encoder, "visual_encoder")
except Exception as e:
    print("error reading visual encoder params:", e)

print("\n-- Q-Former param dtypes (sample):")
try:
    # Qformer may be model.Qformer.bert or model.Qformer
    qformer = getattr(model.Qformer, "bert", model.Qformer)
    sample_param_dtypes(qformer, "qformer")
except Exception as e:
    print("error reading qformer params:", e)

print("\n-- T5 model param dtypes (sample):")
try:
    t5_model = getattr(model, "t5_model", getattr(model, "t5", None))
    if t5_model is not None:
        sample_param_dtypes(t5_model, "t5_model")
except Exception as e:
    print("error reading t5 params:", e)

# === Stage 1: visual encoder (use half dtype input) ===
with torch.no_grad():
    print("\n===== Stage 1: Vision Encoder =====")
    # cast input to half for visual encoder
    image_for_vis = image.to(device).half()
    print(" image_for_vis dtype:", image_for_vis.dtype)

    try:
        image_embeds = model.ln_vision(model.visual_encoder(image_for_vis))
    except Exception as e:
        print("❌ Error when running visual encoder:", e)
        # if conv bias/type mismatch persists, try full float (fallback)
        print("Attempt fallback: cast visual encoder back to float and input to float")
        model.visual_encoder.float()
        model.ln_vision.float()
        try:
            image_embeds = model.ln_vision(model.visual_encoder(image.to(device).float()))
            print("Fallback success (float).")
        except Exception as e2:
            print("Fallback also failed:", e2)
            raise
    print(" ✅ image_embeds shape:", image_embeds.shape, " dtype:", image_embeds.dtype)

    # === Stage 2: Q-Former (Q-Former expects float32 in this installation) ===
    print("\n===== Stage 2: Q-Former =====")
    # cast image_embeds to float32 for qformer if needed
    image_embeds_q = image_embeds.to(torch.float32)
    print(" image_embeds_q dtype:", image_embeds_q.dtype)

    image_atts = torch.ones(image_embeds_q.size()[:-1], dtype=torch.long).to(device)
    query_tokens = model.query_tokens.expand(image_embeds_q.shape[0], -1, -1).to(torch.float32)

    qformer = getattr(model.Qformer, "bert", model.Qformer)
    try:
        query_output = qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_q,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    except Exception as e:
        print("❌ Error when running Q-Former:", e)
        raise
    q_last = query_output.last_hidden_state
    print(" ✅ query_output shape:", q_last.shape, " dtype:", q_last.dtype)

    # === Stage 3: t5_proj -> prepare inputs for T5 ===
    print("\n===== Stage 3: t5_proj -> T5 =====")
    # t5_proj likely float32; compute then cast to T5 dtype (sample: bfloat16)
    inputs_t5 = model.t5_proj(q_last)  # usually float32
    print(" raw inputs_t5 dtype:", inputs_t5.dtype)

    # detect t5 model dtype (if possible) and cast
    t5_model = getattr(model, "t5_model", getattr(model, "t5", None))
    t5_dtype = None
    if t5_model is not None:
        # find first param dtype of t5 model
        for p in t5_model.parameters():
            t5_dtype = p.dtype
            break
    print(" t5_model expected dtype:", t5_dtype)

    if t5_dtype is not None:
        inputs_t5 = inputs_t5.to(t5_dtype)
    print(" final inputs_t5 shape:", inputs_t5.shape, " dtype:", inputs_t5.dtype)

    # try a lightweight T5 forward (with dummy tokens)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
    text_input_ids = torch.zeros((1, 3), dtype=torch.long).to(device)
    try:
        t5_out = t5_model(
            input_ids=text_input_ids,
            encoder_hidden_states=inputs_t5,
            encoder_attention_mask=atts_t5,
            return_dict=True,
        )
        if hasattr(t5_out, "last_hidden_state"):
            print(" ✅ t5 encoder last_hidden_state:", t5_out.last_hidden_state.shape, " dtype:", t5_out.last_hidden_state.dtype)
        else:
            print(" ✅ t5_model returned (no last_hidden_state).")
    except Exception as e:
        print("⚠️ t5_model forward raised:", e)

print("\n🎉 Flow test finished. If all shapes & dtypes look reasonable, proceed to generation.")


