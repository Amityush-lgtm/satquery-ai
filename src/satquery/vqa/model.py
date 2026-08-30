import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
from PIL import Image

from satquery.utils.logging import get_logger

logger = get_logger("satquery.vqa.model")


class BaseVQAModel:
    """Abstract interface for VLM model wrappers."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate_answer(self, image: Image.Image, question: str) -> Tuple[str, Optional[float]]:
        raise NotImplementedError


class MockVQAModel(BaseVQAModel):
    """
    Mock VQA model for testing, offline verification, and CI pipelines.
    Generates realistic remote-sensing answers based on heuristic image inspection.
    """

    def __init__(self, model_id: str = "mock-vlm-v1"):
        super().__init__(model_id)
        logger.info(f"Initialized MockVQAModel (ID: {self.model_id})")

    def generate_answer(self, image: Image.Image, question: str) -> Tuple[str, Optional[float]]:
        w, h = image.size
        # Simple simulated analysis
        q_lower = question.lower()
        if "what" in q_lower or "visible" in q_lower or "describe" in q_lower:
            ans = f"The satellite patch ({w}x{h} px) shows agricultural land, vegetation parcels, and nearby road networks."
        elif "water" in q_lower or "river" in q_lower:
            ans = "No significant open water bodies are detected in this patch."
        elif "urban" in q_lower or "building" in q_lower or "settlement" in q_lower:
            ans = "Scattered rural structures and farm infrastructure are present across the area."
        elif "crop" in q_lower or "forest" in q_lower:
            ans = "The scene exhibits dominant arable land with interspersed deciduous forest canopies."
        else:
            ans = f"In this remote-sensing imagery, mixed land-cover patterns including vegetation and agricultural plots are visible in response to: '{question}'."

        return ans, None


class Qwen2VLModel(BaseVQAModel):
    """
    Wrapper for Qwen2-VL-2B-Instruct vision-language model.
    Optimized for remote sensing VQA with dynamic resolution handling.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
        torch_dtype: str = "float16",
    ):
        super().__init__(model_id)
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info(f"Loading VLM model: {self.model_id} (device={self.device}, dtype={self.torch_dtype})")
        start_t = time.time()

        resolved_device = "cuda" if (self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())) else "cpu"
        dtype = torch.float16 if (resolved_device == "cuda" and self.torch_dtype == "float16") else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        if resolved_device == "cuda":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to("cpu")

        self.model.eval()
        load_time = time.time() - start_t
        logger.info(f"Model loaded successfully in {load_time:.2f}s on {resolved_device}")

    def generate_answer(self, image: Image.Image, question: str) -> Tuple[str, Optional[float]]:
        import torch

        # Format conversation prompt with special image tokens
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Analyze this satellite image and answer the question directly and factually: {question}"},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=False,
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Prototype 1 confidence is null since raw generation logits require uncalibrated softmax
        return output_text, None


_MODEL_INSTANCE: Optional[BaseVQAModel] = None


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads model configuration from YAML."""
    if not config_path:
        config_path = os.getenv("SATQUERY_MODEL_CONFIG", "configs/model.yaml")
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_vqa_model(
    model_id: Optional[str] = None,
    force_mock: bool = False,
    config_path: Optional[str] = None,
) -> BaseVQAModel:
    """
    Singleton factory for acquiring the active VLM instance.
    Loads the model once in memory and reuses it across inferences.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None and not force_mock:
        return _MODEL_INSTANCE

    if force_mock or os.getenv("SATQUERY_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}:
        _MODEL_INSTANCE = MockVQAModel(model_id=model_id or "mock-vlm-v1")
        return _MODEL_INSTANCE

    cfg = load_model_config(config_path)
    selected_id = model_id or cfg.get("selected_model", "Qwen/Qwen2-VL-2B-Instruct")
    runtime_cfg = cfg.get("runtime", {})
    device = os.getenv("SATQUERY_DEVICE", runtime_cfg.get("prefer_device", "auto"))
    torch_dtype = runtime_cfg.get("torch_dtype", "float16")
    mock_fallback = runtime_cfg.get("enable_mock_fallback", True)

    try:
        if "Qwen" in selected_id:
            _MODEL_INSTANCE = Qwen2VLModel(
                model_id=selected_id,
                device=device,
                torch_dtype=torch_dtype,
            )
        else:
            logger.warning(f"Unrecognized model {selected_id}, falling back to Qwen2-VL")
            _MODEL_INSTANCE = Qwen2VLModel(
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                device=device,
                torch_dtype=torch_dtype,
            )
    except Exception as e:
        logger.error(f"Failed to load VLM model '{selected_id}': {e}")
        if mock_fallback:
            logger.warning("Enabling MockVQAModel fallback for testing/offline execution.")
            _MODEL_INSTANCE = MockVQAModel(model_id=f"fallback-mock-({selected_id})")
        else:
            raise

    return _MODEL_INSTANCE
