#!/usr/bin/env python3
"""Shared reference/runtime contract helpers for Krasis reference validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_FINGERPRINT_FILES = (
    "config.json",
    "generation_config.json",
)

TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

TURN_BOUNDARY_TOKENS = (
    "<|im_start|>",
    "<|start_header_id|>",
    "<|begin_of_text|>",
    "<|turn>",
    "<turn|>",
)

REFERENCE_PROFILE_DEFAULT = "greedy_chat_default"
REFERENCE_PROFILE_THINKING_OFF = "greedy_chat_thinking_off"
REFERENCE_PROFILE_THINKING_ON = "greedy_chat_thinking_on"
REFERENCE_PROFILE_UNKNOWN = "greedy_chat_unknown"
REFERENCE_PROFILE_RUNTIME = "runtime_reference_test"

REFERENCE_PROFILES = (
    REFERENCE_PROFILE_DEFAULT,
    REFERENCE_PROFILE_THINKING_OFF,
    REFERENCE_PROFILE_THINKING_ON,
)

PROFILE_FILENAMES = {
    REFERENCE_PROFILE_DEFAULT: "greedy_chat_default.json",
    REFERENCE_PROFILE_THINKING_OFF: "greedy_chat_thinking_off.json",
    REFERENCE_PROFILE_THINKING_ON: "greedy_chat_thinking_on.json",
}

LEGACY_REFERENCE_FILENAMES = (
    "greedy_reference.json",
)

DEFAULT_LOCAL_MODELS_DIR = Path.home() / ".krasis" / "models"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _normalize_extra_special_tokens(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, list):
        return None

    normalized: Dict[str, Any] = {}
    used_keys = set()

    for idx, item in enumerate(raw):
        if isinstance(item, str):
            token_value: Any = item
            key_source = item
        elif isinstance(item, dict):
            token_value = item
            key_source = str(
                item.get("content")
                or item.get("token")
                or item.get("id")
                or f"extra_special_token_{idx}"
            )
        else:
            token_value = item
            key_source = f"extra_special_token_{idx}"

        key = re.sub(r"[^a-z0-9]+", "_", key_source.lower()).strip("_")
        if not key:
            key = f"extra_special_token_{idx}"
        if key[0].isdigit():
            key = f"extra_special_token_{idx}_{key}"
        if not key.endswith("_token"):
            key = f"{key}_token"
        while key in used_keys:
            key = f"{key}_{idx}"
        normalized[key] = token_value
        used_keys.add(key)

    return normalized


def load_tokenizer_with_compat(model_path: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as exc:
        if "object has no attribute 'keys'" not in str(exc):
            raise
        tokenizer_cfg_path = Path(model_path) / "tokenizer_config.json"
        if not tokenizer_cfg_path.is_file():
            raise
        tokenizer_cfg = _load_json(tokenizer_cfg_path)
        normalized = _normalize_extra_special_tokens(tokenizer_cfg.get("extra_special_tokens"))
        if not normalized:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            extra_special_tokens=normalized,
        )
        setattr(tokenizer, "_krasis_extra_special_tokens_compat", normalized)
        return tokenizer


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


EMPTY_TEMPLATE_HASH = _sha256_bytes(b"")
VISIBLE_THOUGHT_RE = re.compile(r"(?is)(<think>|</think>|<\|channel\>|<channel\|>|\bthought\b)")


def _validate_turn_decode_diagnostics(
    turn: Dict[str, Any],
    *,
    expected_steps: int,
    expected_top_k: int,
) -> Optional[str]:
    token_ids = turn.get("token_ids")
    per_token_data = turn.get("per_token_data")
    if not isinstance(token_ids, list):
        return "turn token_ids missing"
    if expected_steps <= 0:
        return None
    if not token_ids:
        return None
    if not isinstance(per_token_data, list):
        return "per_token_data missing"

    expected_count = min(len(token_ids), expected_steps)
    if len(per_token_data) != expected_count:
        return f"expected {expected_count} per_token_data entries, got {len(per_token_data)}"

    for idx, entry in enumerate(per_token_data):
        if not isinstance(entry, dict):
            return f"step {idx} entry is not an object"
        if entry.get("token_id") != token_ids[idx]:
            return f"step {idx} token_id mismatch: entry={entry.get('token_id')} token_ids={token_ids[idx]}"
        if not isinstance(entry.get("rank"), int) or entry["rank"] < 1:
            return f"step {idx} rank missing or invalid"
        top_k = entry.get("top_k")
        if not isinstance(top_k, list) or not top_k:
            return f"step {idx} top_k missing"
        if len(top_k) > expected_top_k:
            return f"step {idx} top_k length {len(top_k)} exceeds expected {expected_top_k}"
        top_k_ids = []
        for top_idx, top_entry in enumerate(top_k):
            if not isinstance(top_entry, dict):
                return f"step {idx} top_k[{top_idx}] is not an object"
            token_id = top_entry.get("token_id")
            log_prob = top_entry.get("log_prob")
            if not isinstance(token_id, int):
                return f"step {idx} top_k[{top_idx}] token_id invalid"
            if not isinstance(log_prob, (int, float)):
                return f"step {idx} top_k[{top_idx}] log_prob invalid"
            top_k_ids.append(token_id)
        if token_ids[idx] not in top_k_ids and entry["rank"] <= len(top_k_ids):
            return f"step {idx} rank={entry['rank']} but selected token missing from top_k"
    return None


def _extract_chat_template(tokenizer_cfg: Dict[str, Any], model_dir: Optional[Path] = None) -> Tuple[str, str]:
    chat_template = tokenizer_cfg.get("chat_template")
    if isinstance(chat_template, str):
        return "default", chat_template
    if isinstance(chat_template, list):
        first_name = "default"
        first_template = ""
        for item in chat_template:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            template = item.get("template")
            if isinstance(name, str) and isinstance(template, str):
                if not first_template:
                    first_name = name
                    first_template = template
                if name == "default":
                    return name, template
        if first_template:
            return first_name, first_template
    if model_dir is not None:
        template_path = model_dir / "chat_template.jinja"
        if template_path.is_file():
            return "chat_template.jinja", template_path.read_text()
    return "fallback", ""


def _collect_eos_ids(config_json: Dict[str, Any], generation_json: Dict[str, Any]) -> List[int]:
    eos_ids: List[int] = []
    seen = set()
    for source in (generation_json, config_json):
        eos = source.get("eos_token_id")
        if eos is None:
            continue
        values = eos if isinstance(eos, list) else [eos]
        for value in values:
            if isinstance(value, int) and value not in seen:
                eos_ids.append(value)
                seen.add(value)
    return eos_ids


def collect_capture_stop_ids(
    tokenizer: Any,
    *,
    model_path: Optional[str] = None,
    config_json: Optional[Dict[str, Any]] = None,
    generation_json: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Collect model-authored stop ids for HF reference capture.

    We prefer ids declared by the checkpoint config/generation_config and then
    augment them with tokenizer-discoverable end-of-turn tokens when present.
    This keeps capture aligned with each model's own stop contract instead of
    relying on a small hardcoded Qwen-only token list.
    """

    config_data = config_json
    generation_data = generation_json
    if model_path:
        model_dir = Path(model_path).expanduser().resolve()
        if config_data is None:
            config_data = _load_json(model_dir / "config.json")
        if generation_data is None:
            gen_path = model_dir / "generation_config.json"
            generation_data = _load_json(gen_path) if gen_path.is_file() else {}
    if config_data is None:
        config_data = {}
    if generation_data is None:
        generation_data = {}

    stop_ids = _collect_eos_ids(config_data, generation_data)
    seen = set(stop_ids)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<turn|>", "<|im_end|>", "<|endoftext|>"):
        raw = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(raw, int):
            continue
        if unknown_id is not None and raw == unknown_id:
            continue
        if raw not in seen:
            stop_ids.append(raw)
            seen.add(raw)
    return stop_ids


def _architecture_summary(config_json: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config_json.get("text_config") if isinstance(config_json.get("text_config"), dict) else config_json
    layer_types = cfg.get("layer_types")
    full_attention_interval = cfg.get("full_attention_interval")
    has_linear_attention = False
    if isinstance(full_attention_interval, int) and full_attention_interval > 0:
        has_linear_attention = True
    if isinstance(layer_types, list) and any(v == "linear_attention" for v in layer_types):
        has_linear_attention = True

    return {
        "architectures": cfg.get("architectures", config_json.get("architectures")),
        "hidden_size": cfg.get("hidden_size"),
        "intermediate_size": cfg.get("intermediate_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "n_routed_experts": cfg.get("n_routed_experts"),
        "full_attention_interval": full_attention_interval,
        "layer_types_hash": _sha256_bytes(json.dumps(layer_types, sort_keys=True).encode()) if layer_types is not None else None,
        "has_linear_attention": has_linear_attention,
        "uses_moe": bool(cfg.get("n_routed_experts")),
    }


def _special_token_ids(tokenizer: Any, supports_enable_thinking: bool) -> Dict[str, Any]:
    unknown_id = getattr(tokenizer, "unk_token_id", None)

    def token_id(token: str) -> Optional[int]:
        raw = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(raw, int):
            return None
        if unknown_id is not None and raw == unknown_id:
            return None
        return raw

    turn_boundary_ids = []
    for token in TURN_BOUNDARY_TOKENS:
        tid = token_id(token)
        if tid is not None:
            turn_boundary_ids.append(tid)

    think_end_id = token_id("</think>") if supports_enable_thinking else None
    return {
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "think_end_token_id": think_end_id,
        "turn_boundary_token_ids": turn_boundary_ids,
    }


def _extract_input_ids(template_out: Any) -> List[int]:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if hasattr(template_out, "input_ids"):
        return template_out.input_ids[0].tolist()
    if torch is not None and isinstance(template_out, torch.Tensor):
        return template_out[0].tolist()
    return list(template_out)


def infer_capture_profile(reference: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    stored = reference.get("capture_settings")
    if isinstance(stored, dict):
        enable_thinking = stored.get("enable_thinking")
        return {
            "source": "stored_capture_settings",
            "add_generation_prompt": bool(stored.get("add_generation_prompt", True)),
            "enable_thinking": enable_thinking,
            "template_mode": stored.get("template_mode", "default"),
            "profile_id": stored.get(
                "profile_id",
                REFERENCE_PROFILE_THINKING_OFF if enable_thinking is False else
                REFERENCE_PROFILE_THINKING_ON if enable_thinking is True else
                REFERENCE_PROFILE_DEFAULT,
            ),
        }

    first_turn = None
    for conv in reference.get("conversations", []):
        for turn in conv.get("turns", []):
            if turn.get("input_token_ids"):
                first_turn = turn
                break
        if first_turn is not None:
            break

    if first_turn is None:
        return {
            "source": "missing_input_token_ids",
            "add_generation_prompt": True,
            "enable_thinking": None,
            "template_mode": "unknown",
            "profile_id": REFERENCE_PROFILE_UNKNOWN,
        }

    messages = [{"role": "user", "content": first_turn.get("prompt", "")}]
    candidates = [
        ("enable_thinking_false", False, {"enable_thinking": False}),
        ("enable_thinking_true", True, {"enable_thinking": True}),
        ("default", None, {}),
    ]
    matches: List[Tuple[str, Optional[bool]]] = []
    for template_mode, enable_thinking, kwargs in candidates:
        try:
            template_out = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            continue
        if _extract_input_ids(template_out) == first_turn.get("input_token_ids", []):
            matches.append((template_mode, enable_thinking))

    if len(matches) == 1:
        template_mode, enable_thinking = matches[0]
        return {
            "source": "inferred_from_input_token_ids",
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
            "template_mode": template_mode,
            "profile_id": REFERENCE_PROFILE_THINKING_OFF if enable_thinking is False else
            REFERENCE_PROFILE_THINKING_ON if enable_thinking is True else
            REFERENCE_PROFILE_DEFAULT,
        }

    if len(matches) > 1:
        return {
            "source": "ambiguous_input_token_ids",
            "add_generation_prompt": True,
            "enable_thinking": None,
            "template_mode": "ambiguous",
            "profile_id": REFERENCE_PROFILE_UNKNOWN,
            "candidate_matches": [mode for mode, _ in matches],
        }

    return {
        "source": "unable_to_infer",
        "add_generation_prompt": True,
        "enable_thinking": None,
        "template_mode": "unknown",
        "profile_id": REFERENCE_PROFILE_UNKNOWN,
    }


def profile_filename(profile_id: str) -> str:
    return PROFILE_FILENAMES.get(profile_id, f"{profile_id}.json")


def canonical_profile_id(tokenizer: Any, requested_profile: Optional[str]) -> str:
    if requested_profile and requested_profile != "auto":
        return requested_profile

    chat_template = getattr(tokenizer, "chat_template", "")
    supports_enable_thinking = isinstance(chat_template, str) and "enable_thinking" in chat_template
    if supports_enable_thinking:
        return REFERENCE_PROFILE_THINKING_OFF
    return REFERENCE_PROFILE_DEFAULT


def capture_settings_for_profile(profile_id: str) -> Dict[str, Any]:
    if profile_id == REFERENCE_PROFILE_THINKING_OFF:
        return {
            "add_generation_prompt": True,
            "template_mode": "enable_thinking_false",
            "enable_thinking": False,
            "profile_id": REFERENCE_PROFILE_THINKING_OFF,
            "source": "reference_profile",
        }
    if profile_id == REFERENCE_PROFILE_THINKING_ON:
        return {
            "add_generation_prompt": True,
            "template_mode": "enable_thinking_true",
            "enable_thinking": True,
            "profile_id": REFERENCE_PROFILE_THINKING_ON,
            "source": "reference_profile",
        }
    return {
        "add_generation_prompt": True,
        "template_mode": "default",
        "enable_thinking": None,
        "profile_id": REFERENCE_PROFILE_DEFAULT,
        "source": "reference_profile",
    }


def capture_settings_for_reference(
    reference: Dict[str, Any],
    reference_contract: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    request = reference_contract.get("request", {})
    stored_profile_id = reference_contract.get("profile_id")
    if stored_profile_id and stored_profile_id != REFERENCE_PROFILE_UNKNOWN:
        settings = capture_settings_for_profile(stored_profile_id)
        if request.get("add_generation_prompt") is not None:
            settings["add_generation_prompt"] = bool(request["add_generation_prompt"])
        if "enable_thinking" in request:
            settings["enable_thinking"] = request.get("enable_thinking")
            if request.get("enable_thinking") is True:
                settings["template_mode"] = "enable_thinking_true"
            elif request.get("enable_thinking") is False:
                settings["template_mode"] = "enable_thinking_false"
            else:
                settings["template_mode"] = "default"
        settings["source"] = "reference_contract"
        return settings
    return infer_capture_profile(reference, tokenizer)


def apply_capture_template(tokenizer: Any, messages: List[Dict[str, str]], capture_settings: Dict[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "add_generation_prompt": bool(capture_settings.get("add_generation_prompt", True)),
    }
    enable_thinking = capture_settings.get("enable_thinking")
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def reference_candidate_filenames(profile_id: Optional[str]) -> List[str]:
    names: List[str] = []
    if profile_id:
        names.append(profile_filename(profile_id))
    else:
        names.extend(profile_filename(pid) for pid in REFERENCE_PROFILES)
    names.extend(LEGACY_REFERENCE_FILENAMES)
    return names


def build_contract(
    model_name: str,
    model_path: str,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool],
    profile_id: str,
    prompt_source_path: Optional[str] = None,
    runtime_name: Optional[str] = None,
    runtime_version: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_dir = Path(model_path).expanduser().resolve()
    config_json = _load_json(model_dir / "config.json")
    generation_json = _load_json(model_dir / "generation_config.json") if (model_dir / "generation_config.json").is_file() else {}
    tokenizer_cfg = _load_json(model_dir / "tokenizer_config.json") if (model_dir / "tokenizer_config.json").is_file() else {}
    template_name, template_source = _extract_chat_template(tokenizer_cfg, model_dir)
    supports_enable_thinking = "enable_thinking" in template_source

    prompt_source = None
    if prompt_source_path:
        prompt_path = Path(prompt_source_path)
        prompt_source = {
            "path": str(prompt_path),
            "sha256": _sha256_file(prompt_path),
        }

    contract = {
        "schema_version": 1,
        "profile_id": profile_id,
        "model": {
            "name": model_name,
            "path": str(model_dir),
            "path_basename": model_dir.name,
            "fingerprints": {
                name: _sha256_file(model_dir / name)
                for name in MODEL_FINGERPRINT_FILES
            },
        },
        "architecture": _architecture_summary(config_json),
        "tokenizer": {
            "tokenizer_class": tokenizer.__class__.__name__,
            "fingerprints": {
                name: _sha256_file(model_dir / name)
                for name in TOKENIZER_FINGERPRINT_FILES
            },
            "template_name": template_name,
            "chat_template_hash": _sha256_bytes(template_source.encode()),
            "chat_template_supports_enable_thinking": supports_enable_thinking,
            "special_token_ids": _special_token_ids(tokenizer, supports_enable_thinking),
        },
        "request": {
            "add_generation_prompt": bool(add_generation_prompt),
            "enable_thinking": enable_thinking,
        },
        "generation": {
            "mode": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": max_new_tokens,
        },
        "stop": {
            "eos_token_ids": _collect_eos_ids(config_json, generation_json),
        },
        "prompt_source": prompt_source,
        "runtime": {
            "name": runtime_name,
            "version": runtime_version,
            "torch_dtype": torch_dtype,
        },
    }
    if extra:
        contract["extra"] = extra
    return contract


def load_or_infer_reference_contract(reference: Dict[str, Any], model_path: str, tokenizer: Any) -> Dict[str, Any]:
    stored = reference.get("contract")
    if isinstance(stored, dict):
        return stored

    capture = infer_capture_profile(reference, tokenizer)
    return {
        "schema_version": 1,
        "profile_id": capture["profile_id"],
        "legacy_inferred": True,
        "model": {
            "name": reference.get("model"),
            "path": reference.get("model_path"),
            "path_basename": os.path.basename(str(reference.get("model_path", "")).rstrip("/")),
            "fingerprints": {},
        },
        "request": {
            "add_generation_prompt": capture["add_generation_prompt"],
            "enable_thinking": capture["enable_thinking"],
        },
        "generation": {
            "mode": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": reference.get("max_new_tokens"),
        },
        "stop": {
            "eos_token_ids": reference.get("eos_token_ids", []),
        },
        "tokenizer": {
            "chat_template_supports_enable_thinking": "enable_thinking" in getattr(tokenizer, "chat_template", ""),
        },
        "runtime": {
            "name": reference.get("runtime"),
            "version": reference.get("runtime_version"),
        },
        "extra": {
            "capture_inference_source": capture.get("source"),
            "candidate_matches": capture.get("candidate_matches"),
        },
    }


def classify_reference_artifact_state(reference_contract: Dict[str, Any]) -> str:
    if not reference_contract.get("legacy_inferred"):
        return "explicit_valid"
    if reference_contract.get("profile_id") == REFERENCE_PROFILE_UNKNOWN:
        return "legacy_invalid"
    return "legacy_inferable"


def build_reference_artifact_metadata(
    reference: Dict[str, Any],
    reference_contract: Dict[str, Any],
    reference_path: Optional[str],
) -> Dict[str, Any]:
    path = os.path.abspath(reference_path) if reference_path else None
    return {
        "path": path,
        "filename": os.path.basename(path) if path else None,
        "format_version": reference.get("format_version"),
        "generated_at": reference.get("generated_at"),
        "state": classify_reference_artifact_state(reference_contract),
        "capture_inference_source": reference_contract.get("extra", {}).get("capture_inference_source"),
    }


def build_reference_sanity_report(
    reference: Dict[str, Any],
    *,
    expected_conversations: int,
    expected_turns: int,
) -> Dict[str, Any]:
    conversations = reference.get("conversations", [])
    actual_conversations = len(conversations)
    actual_turns = sum(len(conv.get("turns", [])) for conv in conversations)
    format_version = reference.get("format_version")
    contract = reference.get("contract", {}) if isinstance(reference.get("contract"), dict) else {}
    tokenizer = contract.get("tokenizer", {}) if isinstance(contract.get("tokenizer"), dict) else {}
    stop = contract.get("stop", {}) if isinstance(contract.get("stop"), dict) else {}
    request = contract.get("request", {}) if isinstance(contract.get("request"), dict) else {}
    decode_diagnostics = reference.get("decode_diagnostics", {}) if isinstance(reference.get("decode_diagnostics"), dict) else {}

    profile_candidates = [
        reference.get("profile_id"),
        reference.get("capture_settings", {}).get("profile_id") if isinstance(reference.get("capture_settings"), dict) else None,
        contract.get("profile_id"),
    ]
    profile_values = sorted({value for value in profile_candidates if isinstance(value, str) and value})
    template_hash = tokenizer.get("chat_template_hash")
    artifact_stop_ids = sorted(set(reference.get("eos_token_ids", []) or []))
    contract_stop_ids = sorted(set(stop.get("eos_token_ids", []) or []))

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add_check(
        "conversation_count",
        actual_conversations == expected_conversations,
        f"expected={expected_conversations} actual={actual_conversations}",
    )
    add_check(
        "turn_count",
        actual_turns == expected_turns,
        f"expected={expected_turns} actual={actual_turns}",
    )
    add_check(
        "profile_id_consistent",
        len(profile_values) == 1,
        f"values={profile_values if profile_values else ['missing']}",
    )
    add_check(
        "template_hash_present",
        isinstance(template_hash, str) and len(template_hash) == 64 and template_hash != EMPTY_TEMPLATE_HASH,
        f"template_hash={template_hash or 'missing'}",
    )
    add_check(
        "stop_ids_present",
        bool(artifact_stop_ids),
        f"artifact_stop_ids={artifact_stop_ids}",
    )
    add_check(
        "stop_ids_match_contract",
        bool(artifact_stop_ids) and artifact_stop_ids == contract_stop_ids,
        f"artifact_stop_ids={artifact_stop_ids} contract_stop_ids={contract_stop_ids}",
    )

    diag_steps = decode_diagnostics.get("captured_steps_per_turn")
    diag_top_k = decode_diagnostics.get("top_k")
    diag_errors: List[Dict[str, Any]] = []
    diagnostics_required = isinstance(format_version, int) and format_version >= 5
    diagnostics_declared = isinstance(diag_steps, int) and diag_steps >= 0 and isinstance(diag_top_k, int) and diag_top_k >= 0
    add_check(
        "decode_diagnostics_declared",
        diagnostics_declared if diagnostics_required else True,
        (
            f"format_version={format_version!r} captured_steps_per_turn={diag_steps!r} "
            f"top_k={diag_top_k!r}"
        ),
    )
    if diagnostics_declared:
        for conv_idx, conv in enumerate(conversations):
            for turn_idx, turn in enumerate(conv.get("turns", [])):
                error = _validate_turn_decode_diagnostics(
                    turn,
                    expected_steps=diag_steps,
                    expected_top_k=diag_top_k,
                )
                if error is not None:
                    diag_errors.append(
                        {
                            "conversation_index": conv_idx,
                            "turn_index": turn_idx,
                            "detail": error,
                        }
                    )
        add_check(
            "decode_diagnostics_valid",
            len(diag_errors) == 0,
            f"invalid_turns={len(diag_errors)}",
        )
    elif diagnostics_required:
        add_check(
            "decode_diagnostics_valid",
            False,
            "decode_diagnostics missing for format_version >= 5",
        )

    thought_hits: List[Dict[str, Any]] = []
    effective_profile = profile_values[0] if len(profile_values) == 1 else None
    thinking_off = (
        effective_profile == REFERENCE_PROFILE_THINKING_OFF
        or request.get("enable_thinking") is False
    )
    if thinking_off:
        for conv_idx, conv in enumerate(conversations):
            for turn_idx, turn in enumerate(conv.get("turns", [])):
                text = turn.get("text")
                if not isinstance(text, str):
                    continue
                if VISIBLE_THOUGHT_RE.search(text):
                    thought_hits.append(
                        {
                            "conversation_index": conv_idx,
                            "turn_index": turn_idx,
                            "preview": text[:160].replace("\n", " "),
                        }
                    )
        add_check(
            "thinking_off_visible_thought_leakage",
            len(thought_hits) == 0,
            f"matches={len(thought_hits)}",
        )

    failed_checks = [check for check in checks if not check["ok"]]
    return {
        "status": "pass" if not failed_checks else "fail",
        "expected": {
            "conversations": expected_conversations,
            "turns": expected_turns,
        },
        "actual": {
            "conversations": actual_conversations,
            "turns": actual_turns,
        },
        "checks": checks,
        "failed_checks": [check["name"] for check in failed_checks],
        "thought_leakage": {
            "count": len(thought_hits),
            "samples": thought_hits[:3],
        },
        "decode_diagnostics": {
            "required": diagnostics_required,
            "declared": diagnostics_declared,
            "captured_steps_per_turn": diag_steps if diagnostics_declared else None,
            "top_k": diag_top_k if diagnostics_declared else None,
            "invalid_turns": diag_errors[:3],
        },
    }


def resolve_reference_model_name(reference: Dict[str, Any]) -> Optional[str]:
    model = reference.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    model_path = reference.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        return os.path.basename(model_path.rstrip("/"))
    return None


def _resolve_model_dir(root: Path, model_name: str) -> Optional[Path]:
    candidate = (root / model_name).resolve()
    if candidate.is_dir():
        return candidate

    folded_name = model_name.casefold()
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == folded_name:
                return child.resolve()
    except FileNotFoundError:
        return None
    return None


def resolve_local_reference_model_path(
    reference: Dict[str, Any],
    model_root: Optional[str] = None,
) -> Optional[str]:
    model_name = resolve_reference_model_name(reference)
    roots: List[Path] = []
    if model_root:
        roots.append(Path(model_root).expanduser())
    roots.append(DEFAULT_LOCAL_MODELS_DIR)

    if model_name:
        for root in roots:
            candidate = _resolve_model_dir(root, model_name)
            if candidate is not None:
                return str(candidate)

    stored_model_path = reference.get("model_path")
    if isinstance(stored_model_path, str) and stored_model_path.strip():
        candidate = Path(stored_model_path).expanduser()
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def parse_config_file(conf_path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if key.strip() == "MODEL_PATH":
                key = "CFG_MODEL_PATH"
            cfg[key.strip()] = value
    return cfg


def _config_bool(cfg: Dict[str, str], key: str, default: bool) -> bool:
    raw = cfg.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


def build_runtime_contract(
    conf_path: str,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    effective_profile_id: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = parse_config_file(conf_path)
    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", ""))
    model_name = os.path.basename(model_path.rstrip("/"))
    config_enable_thinking = _config_bool(cfg, "CFG_ENABLE_THINKING", True)
    profile_id = effective_profile_id or REFERENCE_PROFILE_RUNTIME
    capture_settings = capture_settings_for_profile(profile_id)
    return build_contract(
        model_name=model_name,
        model_path=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        add_generation_prompt=capture_settings["add_generation_prompt"],
        enable_thinking=capture_settings["enable_thinking"],
        profile_id=profile_id,
        runtime_name="krasis_reference_test",
        runtime_version=None,
        torch_dtype=None,
        extra={
            "config_path": os.path.abspath(conf_path),
            "server_default_enable_thinking": config_enable_thinking,
            "attention_quant": cfg.get("CFG_ATTENTION_QUANT"),
            "gpu_expert_bits": cfg.get("CFG_GPU_EXPERT_BITS"),
            "cpu_expert_bits": cfg.get("CFG_CPU_EXPERT_BITS"),
            "kv_dtype": cfg.get("CFG_KV_DTYPE"),
        },
    )


def validate_contracts(reference_contract: Dict[str, Any], runtime_contract: Dict[str, Any]) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    ref_profile = reference_contract.get("profile_id")

    ref_model = reference_contract.get("model", {})
    run_model = runtime_contract.get("model", {})
    if reference_contract.get("profile_id") and runtime_contract.get("profile_id"):
        if reference_contract["profile_id"] != runtime_contract["profile_id"]:
            warnings.append(
                f"profile mismatch: reference={reference_contract['profile_id']} runtime={runtime_contract['profile_id']}"
            )
    if ref_model.get("name") and run_model.get("name") and ref_model.get("name") != run_model.get("name"):
        errors.append(f"model name mismatch: reference={ref_model.get('name')} runtime={run_model.get('name')}")
    if ref_model.get("path_basename") and run_model.get("path_basename") and ref_model.get("path_basename") != run_model.get("path_basename"):
        errors.append(
            f"model path basename mismatch: reference={ref_model.get('path_basename')} runtime={run_model.get('path_basename')}"
        )

    ref_fingerprints = ref_model.get("fingerprints", {})
    run_fingerprints = run_model.get("fingerprints", {})
    for name, ref_hash in ref_fingerprints.items():
        run_hash = run_fingerprints.get(name)
        if ref_hash and run_hash and ref_hash != run_hash:
            errors.append(f"model file hash mismatch for {name}")

    ref_arch = reference_contract.get("architecture", {})
    run_arch = runtime_contract.get("architecture", {})
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts_per_tok",
        "n_routed_experts",
        "full_attention_interval",
        "has_linear_attention",
        "uses_moe",
        "layer_types_hash",
    ):
        ref_value = ref_arch.get(key)
        run_value = run_arch.get(key)
        if ref_value is not None and run_value is not None and ref_value != run_value:
            errors.append(f"architecture mismatch for {key}: reference={ref_value} runtime={run_value}")

    ref_tok = reference_contract.get("tokenizer", {})
    run_tok = runtime_contract.get("tokenizer", {})
    for name, ref_hash in ref_tok.get("fingerprints", {}).items():
        run_hash = run_tok.get("fingerprints", {}).get(name)
        if ref_hash and run_hash and ref_hash != run_hash:
            errors.append(f"tokenizer file hash mismatch for {name}")

    if ref_tok.get("chat_template_hash") and run_tok.get("chat_template_hash"):
        if ref_tok["chat_template_hash"] != run_tok["chat_template_hash"]:
            errors.append("chat template hash mismatch")

    ref_special = ref_tok.get("special_token_ids")
    run_special = run_tok.get("special_token_ids")
    if ref_tok.get("chat_template_supports_enable_thinking") is True and isinstance(ref_special, dict) and ref_special.get("think_end_token_id") is None:
        errors.append("reference contract says template supports enable_thinking but </think> token is missing")
    if run_tok.get("chat_template_supports_enable_thinking") is True and isinstance(run_special, dict) and run_special.get("think_end_token_id") is None:
        errors.append("runtime model says template supports enable_thinking but </think> token is missing")

    ref_request = reference_contract.get("request", {})
    run_request = runtime_contract.get("request", {})
    if ref_request.get("add_generation_prompt") is not None and run_request.get("add_generation_prompt") is not None:
        if ref_request["add_generation_prompt"] != run_request["add_generation_prompt"]:
            errors.append(
                "add_generation_prompt mismatch: "
                f"reference={ref_request['add_generation_prompt']} runtime={run_request['add_generation_prompt']}"
            )

    if ref_request.get("enable_thinking") is not None and run_request.get("enable_thinking") is not None:
        if ref_request["enable_thinking"] != run_request["enable_thinking"]:
            errors.append(
                "enable_thinking mismatch: "
                f"reference={ref_request['enable_thinking']} runtime={run_request['enable_thinking']}"
            )

    if run_request.get("enable_thinking") and not run_tok.get("chat_template_supports_enable_thinking"):
        errors.append("runtime config enables thinking, but the model chat template does not support enable_thinking")
    if ref_request.get("enable_thinking") and not ref_tok.get("chat_template_supports_enable_thinking", False):
        errors.append("reference capture claims thinking enabled, but the model chat template does not support enable_thinking")

    server_default_enable_thinking = runtime_contract.get("extra", {}).get("server_default_enable_thinking")
    if server_default_enable_thinking and not run_tok.get("chat_template_supports_enable_thinking"):
        errors.append("server config enables thinking by default, but the model chat template does not support enable_thinking")

    ref_gen = reference_contract.get("generation", {})
    run_gen = runtime_contract.get("generation", {})
    for key in ("mode", "do_sample", "temperature", "top_k", "top_p", "max_new_tokens"):
        ref_value = ref_gen.get(key)
        run_value = run_gen.get(key)
        if ref_value is not None and run_value is not None and ref_value != run_value:
            errors.append(f"generation mismatch for {key}: reference={ref_value} runtime={run_value}")

    ref_stop = reference_contract.get("stop", {}).get("eos_token_ids")
    run_stop = runtime_contract.get("stop", {}).get("eos_token_ids")
    if ref_stop and run_stop and sorted(set(ref_stop)) != sorted(set(run_stop)):
        errors.append(f"eos_token_ids mismatch: reference={list(ref_stop)} runtime={list(run_stop)}")

    if reference_contract.get("legacy_inferred"):
        warnings.append("reference contract was inferred from legacy metadata; file-level fingerprint checks are unavailable")
        source = reference_contract.get("extra", {}).get("capture_inference_source")
        if source == "ambiguous_input_token_ids":
            warnings.append("legacy reference capture settings are ambiguous; enable_thinking could not be inferred from input_token_ids")
        if ref_profile == REFERENCE_PROFILE_UNKNOWN:
            errors.append(
                "legacy reference profile remains unknown; this reference cannot be used for validation or comparison. "
                "Regenerate it with an explicit --profile."
            )

    return {"errors": errors, "warnings": warnings}


def format_contract_report(reference_contract: Dict[str, Any], runtime_contract: Dict[str, Any], validation: Dict[str, List[str]]) -> List[str]:
    lines = [
        f"reference profile: {reference_contract.get('profile_id', 'unknown')}",
        f"runtime profile: {runtime_contract.get('profile_id', 'unknown')}",
        (
            "reference thinking: "
            f"{reference_contract.get('request', {}).get('enable_thinking')}"
            f", runtime thinking: {runtime_contract.get('request', {}).get('enable_thinking')}"
        ),
        (
            "reference template hash: "
            f"{reference_contract.get('tokenizer', {}).get('chat_template_hash', 'missing')}"
        ),
        (
            "runtime template hash: "
            f"{runtime_contract.get('tokenizer', {}).get('chat_template_hash', 'missing')}"
        ),
        (
            "reference eos ids: "
            f"{reference_contract.get('stop', {}).get('eos_token_ids', 'missing')}"
        ),
        (
            "runtime eos ids: "
            f"{runtime_contract.get('stop', {}).get('eos_token_ids', 'missing')}"
        ),
    ]
    if validation.get("warnings"):
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in validation["warnings"])
    if validation.get("errors"):
        lines.append("errors:")
        lines.extend(f"  - {err}" for err in validation["errors"])
    if ref_profile := reference_contract.get("profile_id"):
        if ref_profile == REFERENCE_PROFILE_UNKNOWN:
            lines.append("action:")
            lines.append("  - regenerate the reference with ./dev generate-reference <model> --profile <explicit_profile>")
    return lines


def _trace_enabled(component: str) -> bool:
    if os.environ.get("KRASIS_TRACE") != "1":
        return False
    raw = os.environ.get("KRASIS_TRACE_COMPONENTS", "").strip()
    if not raw:
        return True
    filters = {part.strip().lower() for part in raw.split(",") if part.strip()}
    component = component.lower()
    if "all" in filters or component in filters:
        return True
    prefix = component.split("_", 1)[0]
    return prefix in filters


def emit_contract_trace(
    context: str,
    reference_contract: Dict[str, Any],
    runtime_contract: Dict[str, Any],
    validation: Dict[str, List[str]],
    reference_artifact: Optional[Dict[str, Any]] = None,
) -> None:
    component = "config_contract"
    if not _trace_enabled(component):
        return

    ref_tok = reference_contract.get("tokenizer", {})
    run_tok = runtime_contract.get("tokenizer", {})
    ref_req = reference_contract.get("request", {})
    run_req = runtime_contract.get("request", {})
    ref_model = reference_contract.get("model", {})
    run_model = runtime_contract.get("model", {})
    reference_artifact = reference_artifact or {}

    if reference_artifact:
        print(
            "[KRASIS-TRACE] event=mark scope=global component=config_contract "
            f"context={context} phase=artifact "
            f"state={reference_artifact.get('state', 'unknown')} "
            f"format_version={reference_artifact.get('format_version', 'unknown')} "
            f"file={reference_artifact.get('filename', 'unknown')} "
            f"capture_inference_source={reference_artifact.get('capture_inference_source', 'none')}",
            file=sys.stderr,
        )

    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=reference "
        f"profile={reference_contract.get('profile_id', 'unknown')} "
        f"model={ref_model.get('path_basename', ref_model.get('name', 'unknown'))} "
        f"thinking={ref_req.get('enable_thinking')} "
        f"template_hash={ref_tok.get('chat_template_hash', 'missing')} "
        f"legacy_inferred={bool(reference_contract.get('legacy_inferred', False))}",
        file=sys.stderr,
    )
    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=runtime "
        f"profile={runtime_contract.get('profile_id', 'unknown')} "
        f"model={run_model.get('path_basename', run_model.get('name', 'unknown'))} "
        f"thinking={run_req.get('enable_thinking')} "
        f"template_hash={run_tok.get('chat_template_hash', 'missing')} "
        f"server_default_thinking={runtime_contract.get('extra', {}).get('server_default_enable_thinking')}",
        file=sys.stderr,
    )
    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=validation "
        f"status={'error' if validation.get('errors') else 'warning' if validation.get('warnings') else 'ok'} "
        f"errors={len(validation.get('errors', []))} "
        f"warnings={len(validation.get('warnings', []))}",
        file=sys.stderr,
    )


def emit_contract_failure_trace(
    context: str,
    reason: str,
    **fields: Any,
) -> None:
    component = "config_contract"
    if not _trace_enabled(component):
        return

    parts = [
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=failure reason={reason}"
    ]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace(" ", "_")
        parts.append(f"{key}={text}")
    print(" ".join(parts), file=sys.stderr)


def emit_reference_inventory_trace(
    context: str,
    summary: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> None:
    component = "config_contract"
    if not _trace_enabled(component):
        return

    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=inventory_summary "
        f"total={summary.get('total', 0)} "
        f"explicit_valid={summary.get('state_counts', {}).get('explicit_valid', 0)} "
        f"legacy_inferable={summary.get('state_counts', {}).get('legacy_inferable', 0)} "
        f"legacy_invalid={summary.get('state_counts', {}).get('legacy_invalid', 0)} "
        f"missing_model={summary.get('missing_model_count', 0)} "
        f"scan_errors={summary.get('scan_error_count', 0)}",
        file=sys.stderr,
    )
    for entry in entries:
        artifact = entry.get("reference_artifact", {})
        print(
            "[KRASIS-TRACE] event=mark scope=global component=config_contract "
            f"context={context} phase=inventory_item "
            f"model={entry.get('model_name', 'unknown')} "
            f"state={artifact.get('state', 'unknown')} "
            f"format_version={artifact.get('format_version', 'unknown')} "
            f"file={artifact.get('filename', 'unknown')} "
            f"profile={entry.get('profile_id', 'unknown')} "
            f"model_resolved={entry.get('model_path_resolved', False)} "
            f"errors={entry.get('error_count', 0)} "
            f"warnings={entry.get('warning_count', 0)}",
            file=sys.stderr,
        )
        status = entry.get("status")
        if status == "missing_model":
            emit_contract_failure_trace(
                context,
                "missing_local_model",
                model=entry.get("model_name", "unknown"),
                file=artifact.get("filename") or Path(entry.get("reference_path", "")).name,
                reference_dir=entry.get("reference_dir"),
            )
        elif status == "scan_error":
            emit_contract_failure_trace(
                context,
                "inventory_scan_error",
                model=entry.get("model_name", "unknown"),
                file=Path(entry.get("reference_path", "")).name if entry.get("reference_path") else None,
            )


def emit_reference_generation_trace(
    context: str,
    contract: Dict[str, Any],
    artifact_path: str,
    *,
    max_new_tokens: Optional[int] = None,
) -> None:
    component = "config_contract"
    if not _trace_enabled(component):
        return

    model = contract.get("model", {})
    request = contract.get("request", {})
    tokenizer = contract.get("tokenizer", {})
    generation = contract.get("generation", {})
    artifact = os.path.abspath(artifact_path)

    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=generation_artifact "
        f"state=explicit_valid "
        f"file={os.path.basename(artifact)} "
        f"path={artifact}",
        file=sys.stderr,
    )
    print(
        "[KRASIS-TRACE] event=mark scope=global component=config_contract "
        f"context={context} phase=generation_contract "
        f"profile={contract.get('profile_id', 'unknown')} "
        f"model={model.get('path_basename', model.get('name', 'unknown'))} "
        f"thinking={request.get('enable_thinking')} "
        f"template_hash={tokenizer.get('chat_template_hash', 'missing')} "
        f"max_new_tokens={max_new_tokens if max_new_tokens is not None else generation.get('max_new_tokens')}",
        file=sys.stderr,
    )
