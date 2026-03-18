from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class LocalVLMAdapter:
    def __init__(self, model_dir: Path, max_new_tokens: int = 32) -> None:
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("gpu_required_for_vlm")
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    def infer(self, prompt: str, image_path: Path) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = output[0][prompt_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()
