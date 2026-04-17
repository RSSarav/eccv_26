"""
finetune_knn.py

OpenVLA-OFT fine-tuning with optional KNN-based attention head selection.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import wandb

import finetune as base_ft
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OProjInputActivationExtractor:
    """Extract per-layer/per-head o_proj input activations at a chosen token index."""

    def __init__(self, vla_model: torch.nn.Module):
        text_model = vla_model.language_model.model
        self.layers = text_model.layers
        self.num_layers = len(self.layers)
        self.num_heads = self.layers[0].self_attn.num_heads
        self.head_dim = self.layers[0].self_attn.head_dim
        self._token_idx = -1
        self._current: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def _hook(_module, module_inputs, _output):
            hidden = module_inputs[0].detach()
            idx = min(max(self._token_idx, 0), hidden.shape[1] - 1)
            self._current[layer_idx] = hidden[0, idx].view(self.num_heads, self.head_dim).cpu()

        return _hook

    def register(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.o_proj.register_forward_hook(self._make_hook(layer_idx)))

    def clear(self) -> None:
        self._current = {}

    def set_token_idx(self, token_idx: int) -> None:
        self._token_idx = token_idx

    def snapshot(self) -> torch.Tensor:
        return torch.stack([self._current[i] for i in range(self.num_layers)], dim=0)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _pick_token_index(labels: torch.Tensor, token_mode: str) -> int:
    valid = torch.nonzero(labels != IGNORE_INDEX, as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        return labels.shape[0] - 1
    if token_mode == "state":
        return max(int(valid[0].item()) - 1, 0)
    if token_mode == "first_action":
        return int(valid[0].item())
    if token_mode == "last_action":
        return int(valid[-1].item())
    raise ValueError(f"Unsupported knn_token_mode: {token_mode}")


def _extract_action_target(
    transformed: Dict[str, Any], rlds_step: Dict[str, Any], action_tokenizer: ActionTokenizer
) -> torch.Tensor:
    labels = transformed.get("labels", None)
    if labels is not None:
        mask = labels > action_tokenizer.action_token_begin_idx
        token_ids = labels[mask].detach().cpu().numpy()
        if token_ids.size > 0:
            decoded = action_tokenizer.decode_token_ids_to_actions(token_ids)
            return torch.as_tensor(decoded, dtype=torch.float32).reshape(-1)

    action_vec = torch.as_tensor(rlds_step["action"], dtype=torch.float32)
    if action_vec.ndim > 1:
        action_vec = action_vec[0]
    return action_vec.reshape(-1)


def _predict_actions_from_neighbors(
    neigh_actions: torch.Tensor, nn_sims: torch.Tensor, weighted_neighbors: bool
) -> torch.Tensor:
    if not weighted_neighbors:
        return neigh_actions.mean(dim=2)

    eps = 1e-8
    dists = 1.0 - nn_sims
    near_mask = dists < 1e-8
    has_near = near_mask.any(dim=2)
    nearest_idx = torch.argmin(dists, dim=2)
    a_dim = neigh_actions.shape[-1]
    nearest_actions = neigh_actions.gather(
        dim=2, index=nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, a_dim)
    ).squeeze(2)

    weights = 1.0 / (dists + eps)
    weights = weights / (weights.sum(dim=2, keepdim=True) + eps)
    weighted_pred = (weights.unsqueeze(-1) * neigh_actions).sum(dim=2)
    return torch.where(has_near.unsqueeze(-1), nearest_actions, weighted_pred)


def _single_head_mse_for_k(
    acts_norm_by_head: torch.Tensor,
    act_targets: torch.Tensor,
    episode_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    k_neighbors: int,
    weighted_neighbors: bool,
    temp_excl_w: int,
) -> torch.Tensor:
    _, n_samples, _ = acts_norm_by_head.shape
    sims = torch.matmul(acts_norm_by_head, acts_norm_by_head.transpose(1, 2))
    eye = torch.eye(n_samples, device=acts_norm_by_head.device, dtype=torch.bool).unsqueeze(0)
    sims = sims.masked_fill(eye, -1e9)

    if torch.unique(episode_ids).numel() > 1:
        same_episode = episode_ids[:, None].eq(episode_ids[None, :]).unsqueeze(0)
        sims = sims.masked_fill(same_episode, -1e9)
    else:
        too_close = (frame_ids[:, None] - frame_ids[None, :]).abs() <= temp_excl_w
        sims = sims.masked_fill(too_close.unsqueeze(0), -1e9)

    k_eff = min(k_neighbors, n_samples - 1)
    topk = torch.topk(sims, k=k_eff, dim=-1, largest=True)
    nn_idx, nn_sims = topk.indices, topk.values
    neigh_actions = act_targets[nn_idx]
    pred_actions = _predict_actions_from_neighbors(neigh_actions, nn_sims, weighted_neighbors)
    target_actions = act_targets.unsqueeze(0).expand(acts_norm_by_head.shape[0], -1, -1)
    return ((pred_actions - target_actions) ** 2).mean(dim=(1, 2))


def _subset_mse_for_k(
    acts: torch.Tensor,
    act_targets: torch.Tensor,
    episode_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    head_ids: List[int],
    k_neighbors: int,
    weighted_neighbors: bool,
    temp_excl_w: int,
) -> float:
    n_samples = acts.shape[0]
    subset = acts[:, head_ids, :].reshape(n_samples, -1)
    subset = F.normalize(subset, dim=-1)
    sims = torch.matmul(subset, subset.transpose(0, 1))
    eye = torch.eye(n_samples, device=acts.device, dtype=torch.bool)
    sims = sims.masked_fill(eye, -1e9)

    if torch.unique(episode_ids).numel() > 1:
        same_episode = episode_ids[:, None].eq(episode_ids[None, :])
        sims = sims.masked_fill(same_episode, -1e9)
    else:
        too_close = (frame_ids[:, None] - frame_ids[None, :]).abs() <= temp_excl_w
        sims = sims.masked_fill(too_close, -1e9)

    k_eff = min(k_neighbors, n_samples - 1)
    topk = torch.topk(sims, k=k_eff, dim=-1, largest=True)
    neigh_actions = act_targets[topk.indices].unsqueeze(0)
    pred_actions = _predict_actions_from_neighbors(neigh_actions, topk.values.unsqueeze(0), weighted_neighbors).squeeze(0)
    return float(((pred_actions - act_targets) ** 2).mean().item())


def select_topk_heads_with_knn(
    vla: torch.nn.Module,
    knn_dataset: EpisodicRLDSDataset,
    batch_transform: RLDSBatchTransform,
    action_tokenizer: ActionTokenizer,
    device_id: int,
    k_neighbors: int,
    number_heads: int,
    max_samples: int,
    token_mode: str,
    weighted_neighbors: bool,
    k_grid: List[int],
    temp_excl_w: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    extractor = OProjInputActivationExtractor(vla)
    extractor.register()
    vla.eval()

    activations: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    episode_ids: List[int] = []
    frame_ids: List[int] = []

    with torch.no_grad():
        sample_count = 0
        for ep_idx, rlds_episode in enumerate(knn_dataset.dataset.as_numpy_iterator()):
            ep_len = int(rlds_episode["action"].shape[0])
            for t in range(ep_len):
                if sample_count >= max_samples:
                    break
                rlds_step = tree_map(lambda x: x[t], rlds_episode)
                transformed = batch_transform(rlds_step)
                input_ids = transformed["input_ids"].unsqueeze(0).to(device_id)
                labels = transformed["labels"]
                pixel_values = transformed["pixel_values"].unsqueeze(0).to(torch.bfloat16).to(device_id)
                attention_mask = torch.ones_like(input_ids).to(device_id)

                extractor.clear()
                extractor.set_token_idx(_pick_token_index(labels, token_mode=token_mode))
                _ = vla(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                activations.append(extractor.snapshot())
                actions.append(_extract_action_target(transformed, rlds_step, action_tokenizer))
                episode_ids.append(ep_idx)
                frame_ids.append(t)
                sample_count += 1
            if sample_count >= max_samples:
                break

    extractor.remove()
    if len(activations) < 2:
        raise ValueError("Need at least 2 samples for KNN head selection.")

    acts = torch.stack(activations, dim=0)
    n_samples, n_layers, n_heads, head_dim = acts.shape
    total_heads = n_layers * n_heads
    acts = acts.view(n_samples, total_heads, head_dim).to(device_id, dtype=torch.float32)
    act_targets = torch.stack(actions, dim=0).to(device_id, dtype=torch.float32)
    ep_ids = torch.tensor(episode_ids, device=device_id, dtype=torch.long)
    frm_ids = torch.tensor(frame_ids, device=device_id, dtype=torch.long)
    acts_norm_by_head = F.normalize(acts, dim=-1).permute(1, 0, 2)

    valid_k_grid = sorted(set(int(k) for k in k_grid if int(k) > 0))
    if not valid_k_grid:
        valid_k_grid = [k_neighbors]

    per_k_subset_mse: Dict[int, float] = {}
    best_k = None
    best_subset_mse = float("inf")
    best_top_head_ids: List[int] = []
    best_mse_per_head: Optional[torch.Tensor] = None

    for k in valid_k_grid:
        mse_per_head = _single_head_mse_for_k(
            acts_norm_by_head, act_targets, ep_ids, frm_ids, k, weighted_neighbors, temp_excl_w
        )
        num_select = min(number_heads, total_heads)
        top_head_ids = torch.topk(-mse_per_head, k=num_select, dim=0).indices.tolist()
        subset_mse = _subset_mse_for_k(
            acts, act_targets, ep_ids, frm_ids, top_head_ids, k, weighted_neighbors, temp_excl_w
        )
        per_k_subset_mse[k] = subset_mse
        if subset_mse < best_subset_mse:
            best_subset_mse = subset_mse
            best_k = k
            best_top_head_ids = top_head_ids
            best_mse_per_head = mse_per_head

    assert best_k is not None and best_mse_per_head is not None
    selected = [(hid // n_heads, hid % n_heads) for hid in best_top_head_ids]
    scores = {f"{hid // n_heads}.{hid % n_heads}": float(best_mse_per_head[hid].item()) for hid in best_top_head_ids}
    scores["_best_k"] = float(best_k)
    scores["_best_subset_mse"] = float(best_subset_mse)
    for k, mse in per_k_subset_mse.items():
        scores[f"_subset_mse_k_{k}"] = float(mse)
    return selected, scores


def _build_lora_targets_for_selected_heads(selected_heads: List[Tuple[int, int]], include_mlp: bool = True) -> List[str]:
    layers = sorted({layer for layer, _ in selected_heads})
    targets = [f"language_model.model.layers.{layer}.self_attn.q_proj" for layer in layers]
    if include_mlp:
        for layer in layers:
            targets.extend(
                [
                    f"language_model.model.layers.{layer}.mlp.gate_proj",
                    f"language_model.model.layers.{layer}.mlp.up_proj",
                    f"language_model.model.layers.{layer}.mlp.down_proj",
                ]
            )
    return targets


def _register_qproj_head_grad_masks(vla: torch.nn.Module, selected_heads: List[Tuple[int, int]]) -> None:
    layer_to_heads: Dict[int, set] = {}
    for layer, head in selected_heads:
        layer_to_heads.setdefault(layer, set()).add(head)

    hidden_size = vla.config.text_config.hidden_size
    num_heads = vla.config.text_config.num_attention_heads
    head_dim = hidden_size // num_heads

    for module_name, module in vla.named_modules():
        if ".self_attn.q_proj" not in module_name:
            continue
        try:
            layer_idx = int(module_name.split("language_model.model.layers.")[1].split(".")[0])
        except Exception:
            continue
        if layer_idx not in layer_to_heads:
            continue
        if not hasattr(module, "lora_B") or "default" not in module.lora_B:
            continue

        b_weight = module.lora_B["default"].weight
        if b_weight.shape[0] != hidden_size:
            continue

        row_mask = torch.zeros(hidden_size, dtype=b_weight.dtype, device=b_weight.device)
        for head_idx in layer_to_heads[layer_idx]:
            start = head_idx * head_dim
            end = min((head_idx + 1) * head_dim, hidden_size)
            row_mask[start:end] = 1.0

        b_weight.register_hook(lambda grad, mask=row_mask: grad * mask.unsqueeze(1))


@dataclass
class FinetuneKNNConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "droid_wipe"
    run_root_dir: Path = Path("runs")
    run_id_override: Optional[str] = None
    run_id_note: Optional[str] = None

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False

    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50
    shuffle_buffer_size: int = 100_000

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    knn: bool = False
    number_heads: int = 20
    knn_neighbors: int = 30
    knn_k_grid: str = "10,20,30,40"
    knn_max_samples: int = 200
    knn_include_mlp: bool = True
    knn_token_mode: str = "state"
    knn_weighted_neighbors: bool = True
    knn_temp_excl_w: int = 30
    knn_qproj_grad_mask: bool = True

    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    wandb_log_freq: int = 10


@draccus.wrap()
def finetune_knn(cfg: FinetuneKNNConfig) -> None:
    assert cfg.use_lora, "KNN-based script currently supports LoRA fine-tuning only."
    assert cfg.knn_token_mode in {"state", "first_action", "last_action"}, "Invalid knn_token_mode."
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion."

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = base_ft.get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    if base_ft.model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = base_ft.snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        base_ft.update_auto_map(cfg.vla_path)
        base_ft.check_model_logic_mismatch(cfg.vla_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    selected_heads: Optional[List[Tuple[int, int]]] = None
    head_scores: Dict[str, float] = {}
    if cfg.knn:
        episodic_dataset = EpisodicRLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=False,
        )
        parsed_k_grid = [int(k.strip()) for k in cfg.knn_k_grid.split(",") if k.strip()]
        if distributed_state.is_main_process:
            selected_heads, head_scores = select_topk_heads_with_knn(
                vla=vla,
                knn_dataset=episodic_dataset,
                batch_transform=batch_transform,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                k_neighbors=cfg.knn_neighbors,
                number_heads=cfg.number_heads,
                max_samples=cfg.knn_max_samples,
                token_mode=cfg.knn_token_mode,
                weighted_neighbors=cfg.knn_weighted_neighbors,
                k_grid=parsed_k_grid,
                temp_excl_w=cfg.knn_temp_excl_w,
            )
            with open(run_dir / "knn_selected_heads.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "selected_heads": selected_heads,
                        "scores": head_scores,
                        "config": {
                            "knn_neighbors": cfg.knn_neighbors,
                            "number_heads": cfg.number_heads,
                            "knn_max_samples": cfg.knn_max_samples,
                            "knn_k_grid": parsed_k_grid,
                            "knn_token_mode": cfg.knn_token_mode,
                            "knn_weighted_neighbors": cfg.knn_weighted_neighbors,
                            "knn_temp_excl_w": cfg.knn_temp_excl_w,
                            "knn_include_mlp": cfg.knn_include_mlp,
                        },
                    },
                    f,
                    indent=2,
                )
        if dist.is_available() and dist.is_initialized():
            obj_list = [(selected_heads, head_scores)]
            dist.broadcast_object_list(obj_list, src=0)
            selected_heads, head_scores = obj_list[0]

    if cfg.use_lora:
        target_modules = "all-linear"
        if cfg.knn and selected_heads:
            target_modules = _build_lora_targets_for_selected_heads(selected_heads, include_mlp=cfg.knn_include_mlp)
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        if cfg.knn and selected_heads and cfg.knn_qproj_grad_mask:
            _register_qproj_head_grad_masks(vla, selected_heads)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    action_head = None
    noisy_action_projector = None
    proprio_projector = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = base_ft.init_module(
            module_class=base_ft.L1RegressionActionHead if cfg.use_l1_regression else base_ft.DiffusionActionHead,
            module_name="action_head",
            cfg=cfg,
            device_id=device_id,
            module_args={
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                **({"num_diffusion_steps_train": cfg.num_diffusion_steps_train} if cfg.use_diffusion else {}),
            },
            to_bf16=True,
        )
    if cfg.use_diffusion:
        noisy_action_projector = base_ft.init_module(
            module_class=base_ft.NoisyActionProjector,
            module_name="noisy_action_projector",
            cfg=cfg,
            device_id=device_id,
            module_args={"llm_dim": vla.module.llm_dim},
            to_bf16=True,
        )
    if cfg.use_proprio:
        proprio_projector = base_ft.init_module(
            module_class=base_ft.ProprioProjector,
            module_name="proprio_projector",
            cfg=cfg,
            device_id=device_id,
            module_args={"in_dim": PROPRIO_DIM, "out_dim": vla.module.llm_dim},
            to_bf16=True,
        )

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if action_head is not None:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if noisy_action_projector is not None:
        trainable_params += [p for p in noisy_action_projector.parameters() if p.requires_grad]
    if proprio_projector is not None:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    num_patches = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1
    if cfg.use_diffusion:
        num_patches += 1

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    recent_metrics = {
        "loss_value": base_ft.deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": base_ft.deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": base_ft.deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": base_ft.deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": base_ft.deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = base_ft.run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )
            (loss / cfg.grad_accumulation_steps).backward()
            for k, v in metrics.items():
                if k in recent_metrics:
                    recent_metrics[k].append(v)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened = base_ft.compute_smoothened_metrics(recent_metrics)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                base_ft.log_metrics_to_wandb(smoothened, "VLA Train", log_step, wandb)
                if cfg.knn and head_scores:
                    wandb.log({"KNN/SelectedHeads": len(selected_heads or [])}, step=log_step)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                base_ft.save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune_knn()

