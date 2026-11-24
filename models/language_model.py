import os
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.llms import LLM
import gc, torch
from typing import Any


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS_DIR = "./models"


def download_model():
    local_path = os.path.join(MODELS_DIR, MODEL_NAME.split("/")[1])

    if not os.path.exists(local_path) or not os.listdir(local_path):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path)

    return model, tokenizer


class LocalQwen(LLM):
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def load(self):
        if self._model is None:
            self._model, self._tokenizer = download_model()

    def unload(self):
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        
    @property
    def _llm_type(self):
        return "local-qwen"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=512)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
