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
            raise ValueError('R4 requires the same frozen Llama-3.1-8B Base files as R3')
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

    def generate_report(self, prompt, max_new_tokens):
        from .json_transport import generate_json_batch
        return generate_json_batch(self, [prompt], max_new_tokens)[0]


def generate_model(model, tokenizer, prompt, max_new_tokens, max_tokens):
    """Greedy plain-completion generation for the frozen Base model protocol."""
    import torch
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if not prompt_ids or len(prompt_ids) + max_new_tokens > max_tokens:
        raise InputTooLong('full generation input plus output allowance exceeds context budget')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos = tokenizer.eos_token_id
    if eos is None: raise ValueError('explicit tokenizer EOS required for auditable report termination')
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)
    new_ids = result[0, len(prompt_ids):].tolist()
    ended = bool(new_ids and new_ids[-1] == eos)
    return dict(text=tokenizer.decode(new_ids, skip_special_tokens=True), prompt_tokens=len(prompt_ids),
                generated_tokens=len(new_ids), finish_reason='stop' if ended else 'length')

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
