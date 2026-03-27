from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class LocalVLMAdapter:
    def __init__(self, model_dir: Path, max_new_tokens: int = 160) -> None:
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
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    def _build_inputs(self, prompt: str, image: Image.Image):
        assert self._processor is not None
        # Qwen VL families (2.5/3.x) expect an explicit image token via chat template.
        if hasattr(self._processor, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                rendered = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self._processor(text=[rendered], images=[image], return_tensors="pt")
            except Exception:
                pass
        return self._processor(text=prompt, images=image, return_tensors="pt")

    def infer(self, prompt: str, image_path: Path, max_new_tokens_override: Optional[int] = None) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        image = Image.open(image_path).convert("RGB")
        inputs = self._build_inputs(prompt, image)
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        max_new_tokens = self.max_new_tokens if max_new_tokens_override is None else int(max_new_tokens_override)
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = output[:, prompt_len:]
        if hasattr(self._processor, "batch_decode"):
            decoded = self._processor.batch_decode(generated, skip_special_tokens=True)
            return str(decoded[0] if decoded else "").strip()
        return self._processor.decode(generated[0], skip_special_tokens=True).strip()
