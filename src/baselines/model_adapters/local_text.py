from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalTextAdapter:
    def __init__(self, model_dir: Path, max_new_tokens: int = 160) -> None:
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    def _render_prompt(self, prompt: str) -> str:
        assert self._tokenizer is not None
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": str(prompt or "")}]
                rendered = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass
        return str(prompt or "")

    def infer(self, prompt: str, max_new_tokens_override: Optional[int] = None) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        rendered_prompt = self._render_prompt(prompt)
        inputs = self._tokenizer(rendered_prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        max_new_tokens = self.max_new_tokens if max_new_tokens_override is None else int(max_new_tokens_override)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output[0][prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
