"""Fine-grained FlexLLMGen placement planning for Qwen3.5/Qwen3.8.

Unlike an Accelerate ``device_map`` strategy, this planner treats the pieces
inside every decoder block as independent stages.  Weight percentages have
the same meaning and ordering as the original FlexLLMGen engine: early weight
bytes live on disk, then CPU, and the tail lives on the compute GPU.  The
midpoint rule is intentionally identical to :func:`flexllmgen.flex_opt.init_weight_list`.

The module has no PyTorch dependency.  It reads safetensors headers directly,
so a complete plan can be inspected before importing CUDA or allocating model
weights.
"""

import dataclasses
import json
import os
import re
import struct
import types
import contextvars
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Device = Union[int, str]
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


@dataclasses.dataclass(frozen=True)
class FlexQwenPolicy:
    """The six placement percentages used by the original FlexLLMGen CLI."""

    weight_gpu_percent: float
    weight_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    activation_gpu_percent: float
    activation_cpu_percent: float

    def __post_init__(self) -> None:
        pairs = (
            ("weight", self.weight_gpu_percent, self.weight_cpu_percent),
            ("cache", self.cache_gpu_percent, self.cache_cpu_percent),
            ("activation", self.activation_gpu_percent, self.activation_cpu_percent),
        )
        for label, gpu, cpu in pairs:
            if gpu < 0 or cpu < 0 or gpu + cpu > 100:
                raise ValueError(
                    "{} GPU/CPU percentages must be non-negative and sum to at most 100".format(label)
                )

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "FlexQwenPolicy":
        if len(values) != 6:
            raise ValueError("FlexLLMGen placement requires six percentages")
        return cls(*[float(value) for value in values])

    @property
    def weight_disk_percent(self) -> float:
        return 100.0 - self.weight_gpu_percent - self.weight_cpu_percent

    @property
    def cache_disk_percent(self) -> float:
        return 100.0 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def activation_disk_percent(self) -> float:
        return 100.0 - self.activation_gpu_percent - self.activation_cpu_percent

    def to_dict(self) -> Mapping[str, float]:
        return {
            "weight_gpu_percent": self.weight_gpu_percent,
            "weight_cpu_percent": self.weight_cpu_percent,
            "weight_disk_percent": self.weight_disk_percent,
            "cache_gpu_percent": self.cache_gpu_percent,
            "cache_cpu_percent": self.cache_cpu_percent,
            "cache_disk_percent": self.cache_disk_percent,
            "activation_gpu_percent": self.activation_gpu_percent,
            "activation_cpu_percent": self.activation_cpu_percent,
            "activation_disk_percent": self.activation_disk_percent,
        }


@dataclasses.dataclass(frozen=True)
class QwenStage:
    """One parameter-owning or synthetic compute stage in a Qwen graph."""

    name: str
    category: str
    group: str
    layer: Optional[int]
    checkpoint_bytes: int
    parameter_names: Tuple[str, ...]
    home: Device
    execution_device: Device
    synthetic: bool = False

    def to_dict(self) -> Mapping[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class QwenFineGrainedPlan:
    model: str
    policy: FlexQwenPolicy
    compute_device: Device
    stages: Tuple[QwenStage, ...]
    strategy: str = "flex_percent"

    @property
    def device_map(self) -> Mapping[str, Device]:
        """Return the explicit map consumed by Transformers/Accelerate.

        Synthetic compute stages have no checkpoint tensor and are therefore
        installed by the runtime rather than included in this mapping.
        """
        result: Dict[str, Device] = {}
        for stage in self.stages:
            if stage.synthetic:
                continue
            for parameter_name in stage.parameter_names:
                module_name, _ = parameter_name.rsplit(".", 1)
                # Normal weight/bias tensors are represented by their owning
                # module. Bare DeltaNet parameters (A_log/dt_bias) do not have
                # a callable module boundary and remain resident on compute.
                # Direct DeltaNet parameters use their parent module key. Child
                # projection keys below remain more specific and override it.
                key = module_name
                result[key] = stage.home
        return result

    @property
    def quantization_load_device_map(self) -> Mapping[str, Device]:
        """Map used while quantizing; disk stages are materialized on CPU first.

        TorchAO quantizes a module before Accelerate has written its disk
        representation.  Sending an unquantized checkpoint tensor straight to
        ``disk`` leaves the module on meta and fails conversion.  The runtime
        replaces these temporary CPU homes with per-stage disk hooks after
        quantization.
        """
        return {
            name: ("cpu" if device == "disk" else device)
            for name, device in self.device_map.items()
        }

    def totals_by_home(self) -> Mapping[str, int]:
        totals = defaultdict(int)
        for stage in self.stages:
            if not stage.synthetic:
                totals[str(stage.home)] += stage.checkpoint_bytes
        return dict(sorted(totals.items()))

    def to_dict(self, include_parameters: bool = False) -> Mapping[str, object]:
        stages = []
        for stage in self.stages:
            row = dict(stage.to_dict())
            if not include_parameters:
                row.pop("parameter_names", None)
            stages.append(row)
        return {
            "model": self.model,
            "architecture": "qwen3_5_hybrid",
            "strategy": self.strategy,
            "compute_device": self.compute_device,
            "policy": self.policy.to_dict(),
            "checkpoint_bytes_by_home": self.totals_by_home(),
            "stages": stages,
        }

    def summary(self) -> Mapping[str, object]:
        categories = defaultdict(lambda: defaultdict(int))
        for stage in self.stages:
            categories[stage.category][str(stage.home)] += 1
        return {
            "architecture": "qwen3_5_hybrid",
            "strategy": self.strategy,
            "compute_device": self.compute_device,
            "policy": self.policy.to_dict(),
            "stage_count": len(self.stages),
            "checkpoint_bytes_by_home": self.totals_by_home(),
            "stage_count_by_category_and_home": {
                category: dict(sorted(homes.items()))
                for category, homes in sorted(categories.items())
            },
        }


def _safetensor_sizes(model_path: str) -> Mapping[str, int]:
    """Read exact tensor byte ranges from all local safetensors headers."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as stream:
            weight_map = json.load(stream)["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(
                "no model.safetensors or model.safetensors.index.json in {}".format(model_path)
            )
        filenames = [os.path.basename(single)]

    sizes: Dict[str, int] = {}
    for filename in filenames:
        path = os.path.join(model_path, filename)
        with open(path, "rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError("invalid safetensors header in {}".format(path))
            header_length = struct.unpack("<Q", raw_length)[0]
            header = json.loads(stream.read(header_length))
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            start, end = metadata["data_offsets"]
            sizes[name] = int(end) - int(start)
    return sizes


def _stage_metadata(parameter_name: str) -> Tuple[str, str, Optional[int]]:
    """Return semantic category, percentage group, and decoder layer."""
    match = _LAYER_RE.match(parameter_name)
    if not match:
        if parameter_name.startswith("model.visual."):
            parts = parameter_name.split(".")
            block = parts[3] if len(parts) > 3 and parts[2] == "blocks" else "global"
            return "vision", "vision:{}".format(block), None
        if "embed_tokens" in parameter_name:
            return "embedding", "embedding", None
        if parameter_name.startswith("lm_head"):
            return "lm_head", "lm_head", None
        return "model", "model", None

    layer = int(match.group(1))
    suffix = match.group(2)
    if suffix.startswith("input_layernorm"):
        category = "input_layernorm"
        family = "attention"
    elif suffix.startswith("post_attention_layernorm"):
        category = "post_attention_layernorm"
        family = "mlp"
    elif suffix.startswith("self_attn."):
        category = "full_attention." + suffix.split(".")[1]
        family = "attention"
    elif suffix.startswith("linear_attn."):
        leaf = suffix.split(".")[1]
        category = "linear_attention." + leaf
        family = "attention"
    elif suffix.startswith("mlp."):
        category = "mlp." + suffix.split(".")[1]
        family = "mlp"
    else:
        category = "layer"
        family = "other"
    return category, "layer:{}:{}".format(layer, family), layer


def _stage_name(parameter_name: str) -> str:
    module_name, leaf = parameter_name.rsplit(".", 1)
    return module_name if leaf in {"weight", "bias"} else parameter_name


def _semantic_order(category: str) -> int:
    order = {
        "input_layernorm": 0,
        "full_attention.q_proj": 10,
        "full_attention.q_norm": 11,
        "full_attention.k_proj": 12,
        "full_attention.k_norm": 13,
        "full_attention.v_proj": 14,
        "linear_attention.in_proj_qkv": 10,
        "linear_attention.in_proj_z": 11,
        "linear_attention.in_proj_b": 12,
        "linear_attention.in_proj_a": 13,
        "linear_attention.conv1d": 14,
        "linear_attention.A_log": 15,
        "linear_attention.dt_bias": 16,
        "linear_attention.norm": 18,
        "full_attention.o_proj": 20,
        "linear_attention.out_proj": 20,
        "post_attention_layernorm": 30,
        "mlp.gate_proj": 40,
        "mlp.up_proj": 41,
        "mlp.down_proj": 42,
    }
    return order.get(category, 100)


def _choose_home(midpoint_percent: float, gpu: float, cpu: float, compute_device: Device) -> Device:
    disk = 100.0 - gpu - cpu
    if midpoint_percent < disk:
        return "disk"
    if midpoint_percent < disk + cpu:
        return "cpu"
    return compute_device


def build_qwen_plan(
    model_path: str,
    policy: FlexQwenPolicy,
    compute_device: Device,
) -> QwenFineGrainedPlan:
    """Build an architecture-aware stage plan from a local Qwen checkpoint."""
    expanded = os.path.abspath(os.path.expanduser(model_path))
    sizes = _safetensor_sizes(expanded)

    combined: Dict[str, Dict[str, object]] = {}
    for parameter_name, size in sizes.items():
        # Qwen3.8 checkpoints include training-time multi-token-prediction
        # tensors, while Qwen3_5ForConditionalGeneration intentionally ignores
        # them at inference. They must not consume placement budget or map keys.
        if parameter_name.startswith("mtp."):
            continue
        category, group, layer = _stage_metadata(parameter_name)
        name = _stage_name(parameter_name)
        entry = combined.setdefault(name, {
            "category": category,
            "group": group,
            "layer": layer,
            "bytes": 0,
            "parameters": [],
        })
        entry["bytes"] = int(entry["bytes"]) + size
        entry["parameters"].append(parameter_name)

    by_group: Dict[str, List[Tuple[str, Dict[str, object]]]] = defaultdict(list)
    for name, entry in combined.items():
        by_group[str(entry["group"])].append((name, entry))

    stages: List[QwenStage] = []
    for group in sorted(by_group):
        entries = sorted(
            by_group[group],
            key=lambda item: (_semantic_order(str(item[1]["category"])), item[0]),
        )
        total = sum(int(entry["bytes"]) for _, entry in entries)
        cumulative = 0
        for name, entry in entries:
            size = int(entry["bytes"])
            midpoint = (cumulative + size / 2.0) * 100.0 / max(total, 1)
            home = _choose_home(
                midpoint,
                policy.weight_gpu_percent,
                policy.weight_cpu_percent,
                compute_device,
            )
            # A_log and dt_bias are read directly by DeltaNet rather than by a
            # child module. They are tiny and must remain on the core execution
            # device; projections and convolution still follow the percentage.
            parameters = tuple(sorted(entry["parameters"]))
            if any(not p.endswith((".weight", ".bias")) for p in parameters):
                home = compute_device
            stages.append(QwenStage(
                name=name,
                category=str(entry["category"]),
                group=group,
                layer=entry["layer"],
                checkpoint_bytes=size,
                parameter_names=parameters,
                home=home,
                execution_device=compute_device,
            ))
            cumulative += size

    # Add the operations which do not own checkpoint tensors. Cache placement
    # determines their execution device, mirroring FlexLLMGen's cpu_cache_compute.
    config_path = os.path.join(expanded, "config.json")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    text_config = config.get("text_config", config)
    layer_types = text_config.get("layer_types", ["full_attention"] * int(text_config["num_hidden_layers"]))
    layer_count = len(layer_types)
    for layer, layer_type in enumerate(layer_types):
        midpoint = (layer + 0.5) * 100.0 / max(layer_count, 1)
        cache_home = _choose_home(
            midpoint,
            policy.cache_gpu_percent,
            policy.cache_cpu_percent,
            compute_device,
        )
        execution = "cpu" if cache_home in {"cpu", "disk"} else compute_device
        category = "linear_attention.delta_rule" if layer_type == "linear_attention" else "full_attention.score"
        stages.append(QwenStage(
            name="model.language_model.layers.{}.{}".format(layer, "delta_rule" if layer_type == "linear_attention" else "attention_score"),
            category=category,
            group="layer:{}:cache".format(layer),
            layer=layer,
            checkpoint_bytes=0,
            parameter_names=(),
            home=cache_home,
            execution_device=execution,
            synthetic=True,
        ))
        activation_home = _choose_home(
            midpoint,
            policy.activation_gpu_percent,
            policy.activation_cpu_percent,
            compute_device,
        )
        stages.append(QwenStage(
            name="model.language_model.layers.{}.hidden_state".format(layer),
            category="activation.hidden_state",
            group="layer:{}:activation".format(layer),
            layer=layer,
            checkpoint_bytes=0,
            parameter_names=(),
            home=activation_home,
            execution_device=("cpu" if activation_home in {"cpu", "disk"} else compute_device),
            synthetic=True,
        ))

    stages.sort(key=lambda stage: (
        -1 if stage.layer is None else stage.layer,
        _semantic_order(stage.category),
        stage.name,
    ))
    return QwenFineGrainedPlan(
        model=expanded,
        policy=policy,
        compute_device=compute_device,
        stages=tuple(stages),
    )


def _move_tensors(value, device):
    """Move tensors nested in the small argument structures used by Qwen."""
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


def _install_full_attention_forward(module, execution_device) -> None:
    """Split Qwen full attention so its score/cache stage can run on CPU."""
    import importlib
    import torch
    import torch.nn.functional as functional

    modeling = importlib.import_module(module.__class__.__module__)
    apply_rope = modeling.apply_rotary_pos_emb
    repeat_kv = modeling.repeat_kv

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = query_states.to(execution_device)
        key_states = key_states.to(execution_device)
        value_states = value_states.to(execution_device)
        cos, sin = _move_tensors(position_embeddings, execution_device)
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(execution_device)
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        # Projection hooks execute streamed weights on the primary GPU.  Gate
        # was deliberately retained there while the score stage ran elsewhere.
        gate_device = gate.device
        attn_output = attn_output.to(gate_device)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights

    module.forward = types.MethodType(forward, module)


_DELTA_DEVICE = contextvars.ContextVar("flexllmgen_qwen_delta_device", default=None)
_PATCHED_DELTA_MODULES = set()


def _install_delta_kernels(module, execution_device) -> None:
    """Route the weightless Qwen DeltaNet rule independently of projections."""
    import importlib

    modeling = importlib.import_module(module.__class__.__module__)
    module_key = id(modeling)
    if module_key not in _PATCHED_DELTA_MODULES:
        for function_name in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            original = getattr(modeling, function_name)

            def routed(*args, __original=original, **kwargs):
                destination = _DELTA_DEVICE.get()
                if destination is not None:
                    args = _move_tensors(args, destination)
                    kwargs = _move_tensors(kwargs, destination)
                return __original(*args, **kwargs)

            setattr(modeling, function_name, routed)
        _PATCHED_DELTA_MODULES.add(module_key)

    original_forward = module.forward

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        return_device = getattr(hidden_states, "device", None)
        token = _DELTA_DEVICE.set(execution_device)
        try:
            output = original_forward(*args, **kwargs)
            return _move_tensors(output, return_device) if return_device is not None else output
        finally:
            _DELTA_DEVICE.reset(token)

    module.forward = types.MethodType(forward, module)


def _install_disk_stage(module, path: str, execution_device) -> None:
    """Persist one already-quantized stage and stream it for every call."""
    import torch
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    from accelerate.utils.modeling import named_module_tensors, set_module_tensor_to_device

    torch_device = "cuda:{}".format(execution_device) if isinstance(execution_device, int) else execution_device

    # Detaching Accelerate's temporary CPU-offload hook materializes its
    # original direct parameters back on CPU before serialization.
    if hasattr(module, "_hf_hook"):
        remove_hook_from_module(module)
    state = {
        name: tensor.detach().cpu()
        for name, tensor in named_module_tensors(module, recurse=False)
    }
    tensor_names = tuple(state)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    del state

    class DiskStageHook(ModelHook):
        no_grad = True

        def init_hook(self, hooked_module):
            for tensor_name in tensor_names:
                set_module_tensor_to_device(hooked_module, tensor_name, "meta")
            return hooked_module

        def pre_forward(self, hooked_module, *args, **kwargs):
            loaded = torch.load(path, map_location=torch_device, weights_only=False)
            for tensor_name, value in loaded.items():
                set_module_tensor_to_device(
                    hooked_module,
                    tensor_name,
                    torch_device,
                    value=value,
                )
            return _move_tensors(args, torch_device), _move_tensors(kwargs, torch_device)

        def post_forward(self, hooked_module, output):
            for tensor_name in tensor_names:
                set_module_tensor_to_device(hooked_module, tensor_name, "meta")
            return output

        def detach_hook(self, hooked_module):
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            for tensor_name, value in loaded.items():
                set_module_tensor_to_device(hooked_module, tensor_name, "cpu", value=value)
            return hooked_module

    add_hook_to_module(module, DiskStageHook())


def install_qwen_fine_grained_runtime(
    model,
    plan: QwenFineGrainedPlan,
    offload_dir: Optional[str] = None,
) -> None:
    """Install architecture-specific compute boundaries after model loading.

    Weight streaming itself is performed by Accelerate hooks generated from
    :attr:`QwenFineGrainedPlan.device_map`.  These patches cover the operations
    without their own modules: softmax attention scores and DeltaNet rules.
    """
    if plan.policy.cache_disk_percent > 0:
        raise NotImplementedError(
            "Qwen fine-grained runtime does not silently emulate disk KV/state cache; "
            "set CACHE_GPU + CACHE_CPU to 100"
        )
    if plan.policy.activation_disk_percent > 0:
        raise NotImplementedError(
            "Qwen fine-grained runtime requires ACT_GPU + ACT_CPU to equal 100"
        )

    modules = dict(model.named_modules())
    disk_stages = [stage for stage in plan.stages if not stage.synthetic and stage.home == "disk"]
    if disk_stages and not offload_dir:
        raise ValueError("fine-grained disk weight placement requires an offload directory")
    for stage in disk_stages:
        target = modules.get(stage.name)
        if target is None:
            raise RuntimeError("Qwen disk stage module is missing: {}".format(stage.name))
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", stage.name) + ".pt"
        _install_disk_stage(
            target,
            os.path.join(os.path.abspath(os.path.expanduser(offload_dir)), "fine-grained", filename),
            plan.compute_device,
        )

    activation_stages = {}
    for stage in plan.stages:
        if not stage.synthetic or stage.layer is None:
            continue
        prefix = "model.language_model.layers.{}".format(stage.layer)
        if stage.category == "full_attention.score":
            target = modules.get(prefix + ".self_attn")
            if target is None:
                raise RuntimeError("Qwen full-attention module is missing at layer {}".format(stage.layer))
            _install_full_attention_forward(target, stage.execution_device)
        elif stage.category == "linear_attention.delta_rule":
            target = modules.get(prefix + ".linear_attn")
            if target is None:
                raise RuntimeError("Qwen linear-attention module is missing at layer {}".format(stage.layer))
            _install_delta_kernels(target, stage.execution_device)
        elif stage.category == "activation.hidden_state":
            activation_stages[stage.layer] = stage

    # Hidden states are only made persistent at decoder-block boundaries.  A
    # CPU assignment therefore mirrors FlexLLMGen's activation home without
    # disrupting residual connections inside the block.
    for layer, stage in activation_stages.items():
        target = modules.get("model.language_model.layers.{}".format(layer))
        if target is None:
            raise RuntimeError("Qwen decoder module is missing at layer {}".format(layer))
        original_forward = target.forward

        def layer_forward(
            self,
            *args,
            __forward=original_forward,
            __device=stage.execution_device,
            __compute=plan.compute_device,
            **kwargs
        ):
            # Move before DecoderLayer captures ``residual = hidden_states``.
            compute = "cuda:{}".format(__compute) if isinstance(__compute, int) else __compute
            args = _move_tensors(args, compute)
            kwargs = _move_tensors(kwargs, compute)
            return _move_tensors(__forward(*args, **kwargs), __device)

        target.forward = types.MethodType(layer_forward, target)
