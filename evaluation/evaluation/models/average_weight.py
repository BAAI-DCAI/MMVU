from transformers import AutoModel, AutoTokenizer
import torch
import copy
# from llava.model.builder import load_pretrained_model

model_path_trained = "/share/liangzy/sft-mllm/output/internvl2-8b/v1-20241027-160249/checkpoint-7055"
model_path_original = "/share/liangzy/model_cache/InternVL2-8B"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
trained_model = AutoModel.from_pretrained(
    model_path_trained,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_map)
original_model = AutoModel.from_pretrained(
    model_path_original,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device_map)


trained_state_dict = trained_model.state_dict()
original_state_dict = original_model.state_dict()


averaged_state_dict = copy.deepcopy(original_state_dict)


for key in averaged_state_dict.keys():
    if key in trained_state_dict:
        # averaged_state_dict[key] = (trained_state_dict[key] + original_state_dict[key]) / 2.0
        averaged_state_dict[key] = (trained_state_dict[key]*0.05) + (original_state_dict[key]*0.95)


averaged_model = copy.deepcopy(trained_model)
averaged_model.load_state_dict(averaged_state_dict)


averaged_model.save_pretrained("/share/liangzy/sft-mllm/averaged_models/internvl_8b_8*2")
