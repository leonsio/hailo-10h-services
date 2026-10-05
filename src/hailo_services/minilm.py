"""Host embeddings and resident Hailo encoder for MiniLM retrieval."""

import logging
from pathlib import Path

import numpy as np

from .config import Settings
from .models import ModelManager

_LOG = logging.getLogger(__name__)



class MiniLM:
    def __init__(self, device, hef_path, manager=None):
        from hailo_platform import FormatType
        from safetensors import safe_open
        from tokenizers import Tokenizer

        directory = Path(hef_path).parent
        manager = manager or ModelManager(Settings())
        tokenizer_path = manager.resolve("minilm-tokenizer", "asset", directory / "minilm-tokenizer.json")
        weights_path = manager.resolve("minilm-weights", "asset", directory / "minilm-model.safetensors")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
        with safe_open(str(weights_path), framework="np") as weights:
            prefix = "embeddings."
            if prefix + "word_embeddings.weight" not in weights.keys():
                prefix = "bert.embeddings."
            self.word = weights.get_tensor(prefix + "word_embeddings.weight")
            self.position = weights.get_tensor(prefix + "position_embeddings.weight")[:128]
            self.segment = weights.get_tensor(prefix + "token_type_embeddings.weight")
            self.gamma = weights.get_tensor(prefix + "LayerNorm.weight")
            self.beta = weights.get_tensor(prefix + "LayerNorm.bias")
        if self.word.shape[1] != 384 or self.position.shape != (128, 384):
            raise ValueError("MiniLM embedding assets have unexpected shapes")
        self.infer_model = device.create_infer_model(str(hef_path))
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.FLOAT32)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        self.config_context = self.infer_model.configure()
        try:
            self.configured = self.config_context.__enter__()
        except BaseException:
            self.config_context = None
            raise
        self.artifacts = {"minilm_tokenizer": str(tokenizer_path), "minilm_weights": str(weights_path)}
        _LOG.info("Loaded resident MiniLM HEF %s on Hailo SHARED device", hef_path)

    def embed(self, text):
        encoded = self.tokenizer.encode(text)
        ids = np.asarray(encoded.ids, dtype=np.intp)
        mask = np.asarray(encoded.attention_mask, dtype=np.float32)
        types = np.asarray(encoded.type_ids, dtype=np.intp)
        hidden = self.word[ids] + self.position + self.segment[types]
        mean = hidden.mean(axis=-1, keepdims=True)
        variance = np.square(hidden - mean).mean(axis=-1, keepdims=True)
        hidden = ((hidden - mean) / np.sqrt(variance + 1e-12)) * self.gamma + self.beta
        frame = np.ascontiguousarray(hidden[np.newaxis, ...], dtype=np.float32)
        output_name = self.infer_model.outputs[0].name
        output = np.empty(self.infer_model.output(output_name).shape, dtype=np.float32)
        bindings = self.configured.create_bindings(output_buffers={output_name: output})
        bindings.input().set_buffer(frame)
        self.configured.wait_for_async_ready(timeout_ms=10000)
        # HailoRT versions differ in whether completion_info is positional or a
        # keyword argument. Accept both; an exception escaping this pybind callback
        # terminates the whole process instead of becoming a Python request error.
        job = self.configured.run_async([bindings], lambda *args, **kwargs: None)
        job.wait(10000)
        hidden_out = bindings.output(output_name).get_buffer().reshape(128, 384)
        vector = (hidden_out * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
        norm = np.linalg.norm(vector)
        return vector / max(float(norm), 1e-12)

    def close(self):
        if self.config_context is not None:
            self.config_context.__exit__(None, None, None)
            self.config_context = None
