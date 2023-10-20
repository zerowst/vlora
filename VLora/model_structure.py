from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (  # noqa: E402
    # LoraModel,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)
import torch
import time
import evaluate

MODEL_ALPACA = 'ziqingyang/chinese-alpaca-2-13b'



model = LlamaForCausalLM.from_pretrained(MODEL_ALPACA)

lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        # target_modules=["query_key_value"],
        target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        # target_modules =  model_args.target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
print(model)

lora_model = get_peft_model(model, lora_config)
print(lora_model)
# print('\n\n')
print(model.named_modules())





