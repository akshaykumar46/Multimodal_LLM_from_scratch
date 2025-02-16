import torch
from transformers import AutoModelForVision2Seq

hf_model = AutoModelForVision2Seq.from_pretrained("paligemma-3b-pt-224")
hf_state_dict = hf_model.state_dict()




from gemma_model import PaliGemmaForConditionalGeneration, PaliGemmaConfig
import json
import os
with open(os.path.join("paligemma-3b-pt-224", "config.json"),"r") as f:
    model_config_file = json.load(f)
    config: PaliGemmaConfig = PaliGemmaConfig(**model_config_file)

model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
my_state_dict = model.state_dict()  # Your model

model_keys = set(my_state_dict.keys())
state_dict_keys = set(hf_state_dict.keys())
print("Missing keys in state_dict:", model_keys - state_dict_keys)
print("Unexpected keys in state_dict:", state_dict_keys - model_keys)
# Check mismatched keys
missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
