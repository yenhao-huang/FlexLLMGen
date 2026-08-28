"""Hugging Face backend for non-OPT models with tensor offloading.

The original FlexLLMGen execution engine is tightly coupled to OPT parameter
names and kernels.  This module is an additive backend for newer Transformers
architectures (including Qwen3.8) that delegates model kernels and checkpoint
loading to Transformers while retaining explicit, reproducible tensor
placement controls.

Heavy optional dependencies are imported only when :meth:`HFOffloadLM.load`
is called.  Configuration/search tools can therefore use the contracts in
this module without importing PyTorch.
"""

import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


DeviceKey = Union[int, str]
DeviceMap = Union[str, Mapping[str, Union[int, str]]]

_DEVICE_MAP_STRATEGIES = {
    "auto", "balanced", "balanced_low_0", "sequential", "cpu"
}
_QUANTIZATION_MODES = {"none", "int8", "int8-torchao"}


def _validate_memory_value(value: str, label: str) -> None:
    """Validate an Accelerate memory value without guessing its unit."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string such as '20GiB'".format(label))
    suffixes = ("B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB")
    if not value.strip().endswith(suffixes):
        raise ValueError("{} must include a byte unit, for example '20GiB'".format(label))


@dataclasses.dataclass(frozen=True)
class HFOffloadConfig:
    """Validated input contract for a Hugging Face offload model.

    ``device_map`` accepts the four Accelerate strategies, ``"cpu"`` for a
    DRAM-only baseline, or an explicit module-to-device mapping.  ``max_memory``
    is passed through to Transformers after its values have been validated.
    """

    model: str
    quantization: str = "int8"
    device_map: DeviceMap = "auto"
    max_memory: Optional[Mapping[DeviceKey, str]] = None
    offload_dir: Optional[str] = None
    dtype: str = "auto"
    local_files_only: bool = False
    cpu_offload: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty Hugging Face ID or local path")
        if self.quantization not in _QUANTIZATION_MODES:
            raise ValueError(
                "quantization must be one of {}".format(sorted(_QUANTIZATION_MODES))
            )
        if self.quantization == "int8" and self.device_map == "cpu":
            raise ValueError(
                "bitsandbytes int8 requires an accelerator device map; "
                "use quantization='none' for the DRAM-only baseline"
            )
        if isinstance(self.device_map, str):
            if self.device_map not in _DEVICE_MAP_STRATEGIES:
                raise ValueError(
                    "device_map must be one of {} or an explicit mapping".format(
                        sorted(_DEVICE_MAP_STRATEGIES)
                    )
                )
        elif not isinstance(self.device_map, Mapping) or not self.device_map:
            raise ValueError("an explicit device_map must not be empty")

        if self.max_memory is not None:
            if not self.max_memory:
                raise ValueError("max_memory must not be empty when provided")
            for device, value in self.max_memory.items():
                if not isinstance(device, (int, str)):
                    raise ValueError("max_memory device keys must be integers or strings")
                _validate_memory_value(value, "max_memory[{!r}]".format(device))

        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")

    @property
    def resolved_device_map(self) -> DeviceMap:
        if self.device_map == "cpu":
            return {"": "cpu"}
        return self.device_map

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        if self.max_memory is not None:
            # JSON object keys are strings; make that conversion deterministic.
            result["max_memory"] = {
                str(key): value for key, value in self.max_memory.items()
            }
        return result

    def model_kwargs(self, transformers_module: Any) -> Dict[str, Any]:
        """Build ``from_pretrained`` kwargs using a Transformers-like module."""
        kwargs: Dict[str, Any] = {
            "device_map": self.resolved_device_map,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
        if self.max_memory is not None and self.device_map != "cpu":
            kwargs["max_memory"] = dict(self.max_memory)
        if self.offload_dir is not None and self.device_map != "cpu":
            kwargs["offload_folder"] = os.path.abspath(
                os.path.expanduser(self.offload_dir)
            )
            kwargs["offload_state_dict"] = True
        if self.quantization == "int8":
            kwargs["quantization_config"] = transformers_module.BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self.cpu_offload,
            )
        elif self.quantization == "int8-torchao":
            try:
                from torchao.quantization import Int8WeightOnlyConfig
            except ImportError as exc:
                raise RuntimeError(
                    "TorchAO int8 requires torchao>=0.15; install flexllmgen[qwen]."
                ) from exc
            kwargs["quantization_config"] = transformers_module.TorchAoConfig(
                # Accelerate 1.14's CPU-offload hook calls ``to(dtype)`` while
                # materializing a layer. TorchAO's v2 Int8Tensor deliberately
                # does not implement that operation, while its v1 affine tensor
                # remains movable across CPU/GPU and stores int8 data plus scales.
                # Pin the interoperable representation until Accelerate handles
                # v2 tensor subclasses without the redundant dtype conversion.
                # Per-row (``group_size=None`` in v1) also supports projection
                # widths such as Qwen3.8 vision MLP's 4304 columns.
                quant_type=Int8WeightOnlyConfig(group_size=None, version=1)
            )
        return kwargs


@dataclasses.dataclass(frozen=True)
class HFBenchmarkResult:
    """Stable JSON-serializable output from one generation benchmark."""

    model: str
    quantization: str
    dtype: str
    requested_device_map: DeviceMap
    requested_max_memory: Mapping[str, str]
    device_map: Mapping[str, Union[int, str]]
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    model_logical_bytes: int
    quantized_parameter_tensors: int
    gpu_peak_memory_bytes: Mapping[str, int]
    generated_text: Optional[List[str]] = None
    fine_grained_placement: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    """Append a complete record to a JSON Lines experiment file."""
    expanded = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(expanded, "a", encoding="utf-8") as output:
        output.write(json.dumps(dict(record), sort_keys=True) + "\n")


class HFOffloadLM:
    """Lazy Transformers model wrapper with explicit placement controls."""

    def __init__(self, config: HFOffloadConfig, fine_grained_plan=None):
        self.config = config
        self.fine_grained_plan = fine_grained_plan
        self.model = None
        self.tokenizer = None
        self._torch = None

    @staticmethod
    def _auto_model_class(transformers_module: Any) -> Any:
        # AutoModelForMultimodalLM is the documented Qwen3.8 entry point in
        # Transformers 5.x.  The ImageText alias keeps the backend usable with
        # earlier releases that already support the same architecture.
        for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
            model_class = getattr(transformers_module, name, None)
            if model_class is not None:
                return model_class
        raise RuntimeError(
            "The installed Transformers release has no multimodal auto-model class; "
            "install flexllmgen[qwen]."
        )

    def load(self) -> "HFOffloadLM":
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Qwen support requires the optional dependencies; "
                "install with `pip install -e '.[qwen]'`."
            ) from exc

        if self.config.offload_dir is not None:
            os.makedirs(
                os.path.abspath(os.path.expanduser(self.config.offload_dir)),
                exist_ok=True,
            )

        model_class = self._auto_model_class(transformers)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model,
            padding_side="left",
            local_files_only=self.config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        load_config = self.config
        if self.fine_grained_plan is not None and self.config.quantization == "int8-torchao":
            load_config = dataclasses.replace(
                self.config,
                device_map=self.fine_grained_plan.quantization_load_device_map,
            )
        self.model = model_class.from_pretrained(
            self.config.model,
            **load_config.model_kwargs(transformers)
        ).eval()
        if self.fine_grained_plan is not None:
            from flexllmgen.qwen_flex import install_qwen_fine_grained_runtime

            install_qwen_fine_grained_runtime(
                self.model,
                self.fine_grained_plan,
                offload_dir=self.config.offload_dir,
            )
            # Report the requested homes, not the temporary CPU homes used
            # while TorchAO materialized disk-bound stages.
            self.model.hf_device_map = dict(self.fine_grained_plan.device_map)
        self._torch = torch
        return self

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self._torch is None:
            raise RuntimeError("load() must be called before generation")

    def _input_device(self) -> Any:
        self._require_loaded()
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        device_map = getattr(self.model, "hf_device_map", {})
        for destination in device_map.values():
            if destination != "disk":
                return self._torch.device(destination)
        return self._torch.device("cpu")

    def encode(self, prompts: Sequence[str]) -> Mapping[str, Any]:
        self._require_loaded()
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must contain at least one string")
        encoded = self.tokenizer(
            list(prompts), padding=True, return_tensors="pt"
        )
        return {key: value.to(self._input_device()) for key, value in encoded.items()}

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 32) -> List[str]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        encoded = self.encode(prompts)
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        input_length = encoded["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def benchmark(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 32,
        warmup_tokens: int = 1,
    ) -> HFBenchmarkResult:
        """Measure end-to-end deterministic generation throughput."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if warmup_tokens < 0:
            raise ValueError("warmup_tokens must not be negative")
        encoded = self.encode(prompts)
        torch = self._torch

        if warmup_tokens:
            with torch.inference_mode():
                self.model.generate(
                    **encoded, max_new_tokens=warmup_tokens, do_sample=False
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        input_length = int(encoded["input_ids"].shape[1])
        generated = int(output_ids.numel() - encoded["input_ids"].numel())
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gpu_peaks = {}
        if torch.cuda.is_available():
            gpu_peaks = {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            }
        raw_device_map = getattr(self.model, "hf_device_map", {"": str(self._input_device())})
        device_map = {str(key): value for key, value in raw_device_map.items()}
        footprint = getattr(self.model, "get_memory_footprint", lambda: 0)()
        requested_device_map = self.config.device_map
        if isinstance(requested_device_map, Mapping):
            requested_device_map = {
                str(key): value for key, value in requested_device_map.items()
            }
        requested_max_memory = {
            str(key): value for key, value in (self.config.max_memory or {}).items()
        }
        quantized_tensors = sum(
            type(parameter).__name__ != "Parameter"
            for parameter in self.model.parameters()
        )

        return HFBenchmarkResult(
            model=self.config.model,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            requested_device_map=requested_device_map,
            requested_max_memory=requested_max_memory,
            device_map=device_map,
            batch_size=len(prompts),
            prompt_tokens=input_length * len(prompts),
            generated_tokens=generated,
            elapsed_seconds=elapsed,
            tokens_per_second=generated / max(elapsed, 1e-12),
            # Tensor subclasses expose their logical dtype/shape to PyTorch, so
            # this is deliberately named as a logical size rather than claiming
            # to be packed physical storage for TorchAO weights.
            model_logical_bytes=int(footprint),
            quantized_parameter_tensors=quantized_tensors,
            gpu_peak_memory_bytes=gpu_peaks,
            generated_text=generated_text,
            fine_grained_placement=(
                self.fine_grained_plan.summary()
                if self.fine_grained_plan is not None else None
            ),
        )
