"""Candidate math ported from archived S2 score_labels; never generates confidence."""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from ..util import digest

class InputTooLong(ValueError):
    pass

class HFScorer:
    def __init__(self, model_path, max_tokens=18432, dtype="bfloat16", device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = Path(model_path)
        if not root.is_dir():
            raise ValueError("local model directory required; automatic downloads disabled")
        from .model_identity import BASE_MODEL_FILES
        file_manifest = model_manifest(root)
        if file_manifest != BASE_MODEL_FILES:
            raise ValueError('V5 requires the same frozen Llama-3.1-8B Base files as R3')
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
            trust_remote_code=False, torch_dtype=getattr(torch, dtype), device_map=device,
            attn_implementation="sdpa").eval()
        self.max_tokens = min(max_tokens, self.model.config.max_position_embeddings)
        self.identity = {"backend": "huggingface", "path": str(root.resolve()), "dtype": dtype,
                         "attention": "sdpa", "max_tokens": self.max_tokens,
                         "files": file_manifest}

    def score(self, prompt, candidates):
        return score_model(self.model, self.tokenizer, prompt, candidates, self.max_tokens)

    def generate_evidence(self, prompt, max_new_tokens, schema, prefix):
        from .json_transport import generate_json
        return generate_json(self, prompt, max_new_tokens, schema, prefix)


def model_manifest(root):
    import hashlib
    output = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".json", ".model", ".safetensors", ".bin", ".txt"):
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b""):
                    h.update(block)
            output[p.relative_to(root).as_posix()] = h.hexdigest()
    if not output:
        raise ValueError("model manifest is empty")
    return output

def score_model(model, tokenizer, prompt, candidates, max_tokens):
    import torch
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuations = [tokenizer(" " + label.strip(), add_special_tokens=False)["input_ids"]
                     for label in candidates]
    if not prompt_ids or any(not ids for ids in continuations):
        raise ValueError("empty prompt or candidate tokenization")
    if len(prompt_ids) + max(map(len, continuations)) > max_tokens:
        raise InputTooLong("prompt plus complete candidate exceeds context budget; no truncation")
    target_device = next(model.parameters()).device
    parameters = inspect.signature(model.forward).parameters
    keep_name = next((n for n in ("logits_to_keep", "num_logits_to_keep") if n in parameters), None)
    values = []
    for ids in continuations:
        full = torch.tensor([prompt_ids + ids], dtype=torch.long, device=target_device)
        kwargs = {"use_cache": False}
        if keep_name:
            kwargs[keep_name] = len(ids)+1
        with torch.inference_mode():
            logits = model(full, **kwargs).logits[0]
        positions = logits[:-1] if keep_name else logits[len(prompt_ids)-1:len(prompt_ids)-1+len(ids)]
        targets = full[0, len(prompt_ids):]
        value = torch.log_softmax(positions.float(), dim=-1).gather(1, targets[:, None]).sum().item()
        if not math.isfinite(value):
            raise ValueError("nonfinite candidate likelihood")
        values.append(float(value))
    return {"logprobs": values, "prompt_tokens": len(prompt_ids),
            "candidate_token_counts": list(map(len, continuations))}
