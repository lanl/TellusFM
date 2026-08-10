import os
from itertools import product

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Configuration
# ---------------------------

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Model information
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Load toke from environment variable
API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
if API_TOKEN is None:
    raise RuntimeError(
        "HUGGINGFACE_API_TOKEN environment variable is not set.\n"
        "Please set it to your Hugging Face access token."
    )

# ---------------------------
# Load model and tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=API_TOKEN,
    device_map=DEVICE,
)

# ---------------------------
# Prompt templates
# ---------------------------

prompt_templates = [
    "{type} expansion in {material_type} under shear-stress in the {direction} direction.",
    "{type} development in {material_type} subjected to shear-stress along the {direction} axis.",
    "{type} proliferation in {material_type} under shear-stress in the {direction} direction.",
    "{type} response in {material_type} to shear-stress in the {direction} orientation.",
    "{type} deformation in {material_type} under shear-stress in the {direction} direction.",
    "{type} fracture in {material_type} due to shear-stress in the {direction} direction.",
    "{type} strain distribution in {material_type} under shear-stress along the {direction} axis.",
    "{type} crack propagation in {material_type} with shear-stress applied in the {direction} direction.",
    "{type} stress concentration in {material_type} under {direction} shear-stress.",
    "{type} material failure in {material_type} caused by shear-stress in the {direction} direction.",
    # Shuffled variants
    "Shear-stress in the {direction} direction causing {type} expansion in {material_type}.",
    "{material_type} experiencing {type} development along the {direction} axis under shear-stress.",
    "Shear-stress applied in the {direction} direction leading to {type} proliferation in {material_type}.",
    "Stress concentration under {direction} shear-stress resulting in {type} behavior in {material_type}.",
    "{material_type} undergoing {type} fracture due to shear-stress in the {direction} direction.",
    "{type} response in {material_type} under shear-stress oriented in the {direction} direction.",
    "{material_type} subject to shear-stress in the {direction} axis exhibiting {type} behavior.",
    "{direction} shear-stress inducing {type} deformation in {material_type}.",
    "Under shear-stress in the {direction} direction, {material_type} shows {type} characteristics.",
    "{material_type} subjected to {type} dynamics when shear-stress is applied in the {direction} orientation."
]

# ---------------------------
# Hook for self-attention activations
# ---------------------------

self_attn_activations = []

def save_self_attn_activations(module, input_t, output):
    """Hook to capture self-attention activations."""
    self_attn_activations.append(output[0][0, -1, :])

layers_to_hook = [0, 7, 15, -1]

for layer_num in layers_to_hook:
    model.model.layers[layer_num].self_attn.register_forward_hook(save_self_attn_activations)

# ---------------------------
# Parameters
# ---------------------------

material_types = ['steel', 'pbx', 'aluminum', 'shale', 'tungsten']
fracture_types = ["phase-field method", "T-fracture rule-based mode", "X-fracture rule-based mode"]
directions = ["horizontal", "combined", "vertical"]

# ---------------------------
# Generate embeddings for each prompt
# ---------------------------

os.makedirs("emb_folder", exist_ok=True)

for i, template in enumerate(prompt_templates[:1]):  # Example: only first template
    for fracture, direction, material in product(fracture_types, directions, material_types):
        self_attn_activations = []

        # Fill in the prompt template
        prompt = template.format(type=fracture, direction=direction, material_type=material)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at summarizing complex information into a single token. "
                    "Focus on terms describing fracture types, materials, and shear-stress directions."
                )
            },
            {"role": "user", "content": f"{prompt}\n"}
        ]

        print(messages)

        # Tokenize the input using chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate summary token
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
            attention_mask=torch.ones_like(input_ids)
        )

        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True), "\n")

        # Save self-attention embeddings
        embeddings = torch.stack(self_attn_activations)
        torch.save(embeddings, f'emb_folder/{material}_{fracture}_{direction}_{i}.pt')
