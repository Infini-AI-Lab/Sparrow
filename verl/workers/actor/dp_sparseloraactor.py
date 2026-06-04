# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits, logits_to_topk, sum_min_from_topk
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelSparseLoraActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelSparseLoraActor(BasePPOActor): 
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, 
                 config: ActorConfig, 
                 actor_module: nn.Module, 
                 actor_optimizer_lora: torch.optim.Optimizer = None, 
                 actor_optimizer_full: torch.optim.Optimizer = None, 
    ): 
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module 
        self.actor_optimizer_lora = actor_optimizer_lora 
        self.actor_optimizer_full = actor_optimizer_full 
        role = "Ref" if actor_optimizer_full is None else "Actor" 

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name() 
        
        assert self.actor_optimizer_lora is not None and self.actor_optimizer_full is not None, "actor_optimizer_lora and actor_optimizer_full must be provided" 
        self._assert_optimizer_parameterdisjoint() 
        
        self.dense_kl_loss = self.config.dense_kl_loss 
        self.dense_kl_loss_coef = self.config.dense_kl_loss_coef 
        self.dense_kl_loss_type = self.config.dense_kl_loss_type 
    
    def _assert_param_disjoint(self):
        r_params = list(self.actor_sparse_module.parameters())
        u_params = list(self.actor_dense_module.parameters())

        # 1) object identity overlap
        overlap = {id(p) for p in r_params} & {id(p) for p in u_params}
        assert not overlap, "actor_sparse_module and actor_dense_module share parameter objects."

        # 2) storage overlap (tied weights)
        def storages(ps):
            try:
                return {p.storage().data_ptr() for p in ps if p.requires_grad}
            except Exception:
                # FSDP may wrap; unwrap if needed
                return {p.untyped_storage().data_ptr() for p in ps if p.requires_grad}

        rs, us = storages(r_params), storages(u_params)
        assert rs.isdisjoint(us), "Modules share underlying storages (tied weights). Make a real deep copy." 
    
    def _assert_optimizer_parameterdisjoint(self): 
        lora_params = []
        for group in self.actor_optimizer_lora.param_groups:
            lora_params.extend(group["params"])
        full_params = []
        for group in self.actor_optimizer_full.param_groups:
            full_params.extend(group["params"])

        lora_ids = {id(p) for p in lora_params}
        full_ids = {id(p) for p in full_params}
        assert lora_ids.isdisjoint(full_ids), "actor_optimizer_lora and actor_optimizer_full share parameter objects."

        def storages(ps):
            try:
                return {p.untyped_storage().data_ptr() for p in ps}
            except Exception:
                return {p.storage().data_ptr() for p in ps}

        lora_storages = storages(lora_params)
        full_storages = storages(full_params)
        assert lora_storages.isdisjoint(full_storages), (
            "actor_optimizer_lora and actor_optimizer_full share underlying storages (tied weights)."
        )

    def _forwardsparse_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        topk_idx = None
        topk_logp = None
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True 

                for module in self.actor_module.modules(): 
                    if "attention" in module.__class__.__name__.lower(): 
                        module.use_full_dense = False # TODO: change to False for the update model 
                        # module.use_full_dense = True 

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating 

                if self.use_fused_kernels:
                    assert not self.config.use_double_min_sample, (
                        "This sampling strategy isn't currently supported for fused kernels"
                    )
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if not self.actor_optimizer_lora is None: 
                        if self.config.use_double_min_sample:
                            topk_idx, topk_logp, _ = logits_to_topk(logits_rmpad, self.config.log_probs_to_keep) 

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if not self.actor_optimizer_lora is None: 
                        if self.config.use_double_min_sample:
                            topk_idx = gather_outputs_and_unpad(
                                topk_idx,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                            topk_logp = gather_outputs_and_unpad(
                                topk_logp,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if not self.actor_optimizer_lora is None: 
                    if self.config.use_double_min_sample:
                        full_topk_idx = pad_input(
                            hidden_states=topk_idx.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_topk_logp = pad_input(
                            hidden_states=topk_logp.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )


                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if not self.actor_optimizer_lora is None: 
                    if self.config.use_double_min_sample:
                        topk_idx = full_topk_idx.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)
                        topk_logp = full_topk_logp.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True 
                
                for module in self.actor_module.modules():
                    if "attention" in module.__class__.__name__.lower(): 
                        module.use_full_dense = False 

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating 

                if self.use_fused_kernels:
                    assert not self.config.use_double_min_sample, (
                        "This sampling strategy isn't currently supported for fused kernels"
                    )
                        
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if not self.actor_optimizer_lora is None: 
                        if self.config.use_double_min_sample:
                            topk_idx, topk_logp, _ = logits_to_topk(logits, self.config.log_probs_to_keep)
                    
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, topk_idx, topk_logp 

    def _forwarddense_micro_batch( 
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        topk_idx = None
        topk_logp = None
        response_length = micro_batch["responses"].size(-1) 
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding: 
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True 
                
                for module in self.actor_module.modules(): 
                    if "attention" in module.__class__.__name__.lower(): 
                        module.use_full_dense = True 

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating 

                if self.use_fused_kernels:
                    assert not self.config.use_double_min_sample, (
                        "This sampling strategy isn't currently supported for fused kernels"
                    )
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else: 
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if not self.actor_optimizer_full is None: 
                        if self.config.use_double_min_sample:
                            topk_idx, topk_logp, _ = logits_to_topk(logits_rmpad, self.config.log_probs_to_keep) 

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if not self.actor_optimizer_full is None: 
                        if self.config.use_double_min_sample:
                            topk_idx = gather_outputs_and_unpad(
                                topk_idx,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                            topk_logp = gather_outputs_and_unpad(
                                topk_logp,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if not self.actor_optimizer_full is None: 
                    if self.config.use_double_min_sample:
                        full_topk_idx = pad_input(
                            hidden_states=topk_idx.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_topk_logp = pad_input(
                            hidden_states=topk_logp.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )


                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if not self.actor_optimizer_full is None: 
                    if self.config.use_double_min_sample:
                        topk_idx = full_topk_idx.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)
                        topk_logp = full_topk_logp.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)

            else:  # not using rmpad and no ulysses sp 
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating 

                if self.use_fused_kernels:
                    assert not self.config.use_double_min_sample, (
                        "This sampling strategy isn't currently supported for fused kernels"
                    )
                        
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size) 
                    if not self.actor_optimizer_full is None:
                        if self.config.use_double_min_sample:
                            topk_idx, topk_logp, _ = logits_to_topk(logits, self.config.log_probs_to_keep)
                    
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, topk_idx, topk_logp 

    def _optimizer_lora_step(self): 
        assert self.config.grad_clip is not None
        lora_params = [p for g in self.actor_optimizer_lora.param_groups for p in g["params"]]
        if not lora_params:
            grad_norm = torch.tensor(0.0, device=get_device_id())
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(lora_params, max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=self.config.grad_clip)
        grads_present = sum(1 for p in lora_params if p.grad is not None)
        # print("********** LoRA grads present: %s/%s **************" % (grads_present, len(lora_params))) 

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            # print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}") 
            self.actor_optimizer_lora.zero_grad() 
        else:
            self.actor_optimizer_lora.step() 
        return grad_norm 

    def _optimizer_full_step(self): 
        assert self.config.grad_clip is not None
        full_params = [p for g in self.actor_optimizer_full.param_groups for p in g["params"]]
        if not full_params:
            grad_norm = torch.tensor(0.0, device=get_device_id())
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(full_params, max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(full_params, max_norm=self.config.grad_clip) 
        grads_present = sum(1 for p in full_params if p.grad is not None)
        print("******** Full grads present: %s/%s **************" % (grads_present, len(full_params))) 

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer_full.zero_grad() 
        else:
            self.actor_optimizer_full.step() 
        return grad_norm 

    @GPUMemoryLogger(role="dp actor", logger=logger) 
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor: 
        raise NotImplementedError("compute_log_prob is not implemented for DataParallelMultipleActors") 

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def computesparse_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval() 

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_idx_lst = []
        topk_logp_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_idx, topk_logp = self._forwardsparse_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
                
            log_probs_lst.append(log_probs)
            if not self.actor_optimizer_lora is None:
                if self.config.use_double_min_sample:
                    if not topk_idx is None:
                        topk_idx_lst.append(topk_idx)
                        topk_logp_lst.append(topk_logp)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if not self.actor_optimizer_lora is None:
            if self.config.use_double_min_sample:
                if topk_idx_lst:
                    topk_idx = torch.concat(topk_idx_lst, dim=0)
                    topk_logp = torch.concat(topk_logp_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if not self.actor_optimizer_lora is None:
                if self.config.use_double_min_sample:
                    if (topk_idx_lst):
                        topk_idx = restore_dynamic_batch(topk_idx, batch_idx_list)
                        topk_logp = restore_dynamic_batch(topk_logp, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
        return log_probs, entropys, topk_idx, topk_logp 

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def computedense_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval() 

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_idx_lst = []
        topk_logp_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad(): 
                with self.actor_module.disable_adapter(): 
                    entropy, log_probs, topk_idx, topk_logp = self._forwarddense_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    ) 
                
            log_probs_lst.append(log_probs)
            if not self.actor_optimizer_full is None:
                if self.config.use_double_min_sample:
                    if not topk_idx is None:
                        topk_idx_lst.append(topk_idx)
                        topk_logp_lst.append(topk_logp)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if not self.actor_optimizer_full is None:
            if self.config.use_double_min_sample:
                if topk_idx_lst:
                    topk_idx = torch.concat(topk_idx_lst, dim=0)
                    topk_logp = torch.concat(topk_logp_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if not self.actor_optimizer_full is None:
                if self.config.use_double_min_sample:
                    if (topk_idx_lst):
                        topk_idx = restore_dynamic_batch(topk_idx, batch_idx_list)
                        topk_logp = restore_dynamic_batch(topk_logp, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
        return log_probs, entropys, topk_idx, topk_logp 

    def set_requires_grad(self, opt, flag: bool): 
        for g in opt.param_groups:
            for p in g["params"]:
                p.requires_grad_(flag) 

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "oldsparse_log_probs", 
            "olddense_log_probs", 
            "advantages",
        ] 
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")
        if self.config.use_double_min_sample:
            assert "rollout_log_probs_topk_logprob" in data.batch.keys() and "rollout_log_probs_topk_idx" in data.batch.keys(), (
                "Double Min Importance Smpling requires top-k logits"
                "requires to configure `actor_rollout_ref.rollout.calculate_log_probs=True` "
            )
            select_keys.append("rollout_log_probs")
            select_keys.append("rollout_log_probs_topk_logprob")
            select_keys.append("rollout_log_probs_topk_idx")
            select_keys.append("olddense_topk_idx") 
            select_keys.append("olddense_topk_logp") 
            

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer_lora.zero_grad() 
                self.actor_optimizer_full.zero_grad() 

                cached_olddense = [] 
                self.set_requires_grad(self.actor_optimizer_lora, False)
                self.set_requires_grad(self.actor_optimizer_full, True)
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]

                    # second update the update model weights
                    ########
                    olddense_log_prob = model_inputs["olddense_log_probs"]
                    olddense_topk_idx = model_inputs["olddense_topk_idx"] if self.config.use_double_min_sample else None
                    olddense_topk_logp = model_inputs["olddense_topk_logp"] if self.config.use_double_min_sample else None
                    rollout_log_probs = (
                        model_inputs["rollout_log_probs"]
                        if self.config.tis_imp_ratio_cap > 0 or self.config.use_double_min_sample
                        else None
                    )

                    rollout_log_probs_topk_idx = (
                        model_inputs["rollout_log_probs_topk_idx"] if self.config.use_double_min_sample else None
                    )
                    rollout_log_probs_topk_logprob = (
                        model_inputs["rollout_log_probs_topk_logprob"] if self.config.use_double_min_sample else None
                    )

                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True 

                    with self.actor_module.disable_adapter():
                        entropy_update, log_prob_update, topk_idx_update, topk_logp_update = self._forwarddense_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        if on_policy:
                            olddense_log_prob = log_prob_update.detach()
                            if self.config.use_double_min_sample and not self.config.double_min_apply_trust_region:
                                olddense_topk_idx = topk_idx_update.detach()
                                olddense_topk_logp = topk_logp_update.detach()
                        else:
                            if self.config.use_double_min_sample:
                                if self.config.double_min_use_latest_logits:
                                    if not self.config.double_min_apply_trust_region:
                                        olddense_log_prob = log_prob_update.detach()
                                    else:
                                        olddense_log_prob = model_inputs["olddense_log_probs"]
                                    olddense_topk_idx = topk_idx_update.detach()
                                    olddense_topk_logp = topk_logp_update.detach()
                                else:
                                    olddense_log_prob = model_inputs["olddense_log_probs"]
                                    olddense_topk_idx = model_inputs["olddense_topk_idx"]
                                    olddense_topk_logp = model_inputs["olddense_topk_logp"]
                            else:
                                if self.config.tis_use_new_ckpt_debug:
                                    olddense_log_prob = log_prob_update.detach()
                                else:
                                    olddense_log_prob = model_inputs["olddense_log_probs"]

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        print("######### loss_mode #########", loss_mode)
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        if self.config.use_double_min_sample:
                            pg_loss_update, pg_clipfrac_update, ppo_kl_update, pg_clipfrac_lowerupdate, acceptance_rate_update, ppo_kl_vllm_update, ppo_kl_double_mindense = policy_loss_fn(
                                old_log_prob=olddense_log_prob,
                                log_prob=log_prob_update,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_log_probs=rollout_log_probs,
                                rollout_log_probs_topk_idx=rollout_log_probs_topk_idx,
                                rollout_log_probs_topk_logprob=rollout_log_probs_topk_logprob,
                                old_topk_idx=olddense_topk_idx,
                                old_topk_logp=olddense_topk_logp,
                            )
                        else:
                            pg_loss_update, pg_clipfrac_update, ppo_kl_update, pg_clipfrac_lowerupdate = policy_loss_fn(
                                old_log_prob=olddense_log_prob,
                                log_prob=log_prob_update,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_log_probs=rollout_log_probs,
                            )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(
                                loss_mat=entropy_update, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                            )

                            # compute policy loss
                            policy_loss_update = pg_loss_update - entropy_loss * entropy_coeff
                        else:
                            policy_loss_update = pg_loss_update

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob_update,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss_update = policy_loss_update + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss_update"] = kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef_update"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss_update = policy_loss_update * loss_scale_factor
                        else:
                            loss_update = policy_loss_update * loss_scale_factor
                        full_params = [p for g in self.actor_optimizer_full.param_groups for p in g["params"]]
                        full_requires = sum(1 for p in full_params if p.requires_grad)
                        print(
                            f"*** Full requires_grad before backward: {full_requires}/{len(full_params)} ***",
                            flush=True,
                        )
                        loss_update.backward()
                        full_grads = sum(1 for p in full_params if p.grad is not None)
                        print(
                            f"*** Full grads after backward: {full_grads}/{len(full_params)} ***",
                            flush=True,
                        )

                    metrics_dict = {
                        "actor/pg_loss_update": pg_loss_update.detach().item() * loss_scale_factor,
                        "actor/pg_clipfrac_update": pg_clipfrac_update.detach().item(),
                        "actor/ppo_kl_update": ppo_kl_update.detach().item(),
                        "actor/pg_clipfrac_lower_update": pg_clipfrac_lowerupdate.detach().item(),
                    }
                    if self.config.use_double_min_sample:
                        double_min_sample_metrics_dict = {  # double min sample is only used in the update model
                            "actor/sampler_acceptance_rate_update": acceptance_rate_update,
                            "actor/ppo_kl_vllm_update": ppo_kl_vllm_update.detach().item(),
                            "actor/ppo_kl_double_mindense": ppo_kl_double_mindense.detach().item(),
                        }

                        metrics_dict.update(double_min_sample_metrics_dict)

                    micro_batch_metrics.update(metrics_dict)
                    append_to_dict(metrics, micro_batch_metrics)

                    cached_olddense.append(
                        {
                            "olddense_log_probs": olddense_log_prob.detach().cpu().pin_memory(),
                            "olddense_topk_idx": (
                                olddense_topk_idx.detach().cpu().pin_memory()
                                if olddense_topk_idx is not None
                                else None
                            ),
                            "olddense_topk_logp": (
                                olddense_topk_logp.detach().cpu().pin_memory()
                                if olddense_topk_logp is not None
                                else None
                            ),
                        }
                    )

                grad_norm = self._optimizer_full_step()
                mini_batch_metrics = {"actor/grad_norm_update": grad_norm.detach().item(), "actor/stop_update_model": 0}
                append_to_dict(metrics, mini_batch_metrics)
                self.actor_optimizer_full.zero_grad() 
                
                self.actor_optimizer_lora.zero_grad() 

                self.set_requires_grad(self.actor_optimizer_lora, True)
                self.set_requires_grad(self.actor_optimizer_full, False)
                for micro_batch, cached in zip(micro_batches, cached_olddense):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]

                    model_inputs["olddense_log_probs"] = cached["olddense_log_probs"].to(
                        get_device_id(), non_blocking=True
                    )
                    if cached["olddense_topk_idx"] is not None:
                        model_inputs["olddense_topk_idx"] = cached["olddense_topk_idx"].to(
                            get_device_id(), non_blocking=True
                        )
                    if cached["olddense_topk_logp"] is not None:
                        model_inputs["olddense_topk_logp"] = cached["olddense_topk_logp"].to(
                            get_device_id(), non_blocking=True
                        )

                    # first update the rollout model weights
                    ######## 
                    # rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None 
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.sparse_tis_imp_ratio_cap > 0 else None 
                    advantages = model_inputs["advantages"] 

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation 

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True 

                    entropy, log_prob, topk_idx, topk_logp = self._forwardsparse_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    ) 
                    
                    if on_policy:
                        oldsparse_log_prob = log_prob.detach() 
                    else:
                        oldsparse_log_prob = model_inputs["oldsparse_log_probs"] 
                    
                    if not self.config.disable_policy_gradient: 
                        policy_loss_fn = get_policy_loss_fn("vanilla_old") 
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=oldsparse_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_log_probs=rollout_log_probs,
                        ) 
                        
                        policy_loss = pg_loss 
                        
                        metrics_dict = {
                            "actor/pg_rollout_sparseloss": pg_loss.detach().item() * loss_scale_factor, 
                            "actor/pg_rollout_sparselclipfrac": pg_clipfrac.detach().item(), 
                            "actor/ppo_rollout_sparselkl": ppo_kl.detach().item(), 
                            "actor/pg_rollout_sparselclipfrac_lower": pg_clipfrac_lower.detach().item(), 
                        } 
                        
                        micro_batch_metrics.update(metrics_dict) 
                        append_to_dict(metrics, micro_batch_metrics) 

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov 

                    # print("######### bigupdate_kl_loss #########") 
                    if self.dense_kl_loss: 
                        update_log_prob = model_inputs["olddense_log_probs"].detach()
                        klloss = kl_penalty(
                            logprob=log_prob, ref_logprob=update_log_prob, kl_penalty=self.dense_kl_loss_type
                        )

                        kl_loss = agg_loss(loss_mat=klloss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics.update(
                            {
                                "actor/kl_loss_bigupdate": kl_loss.detach().item() * loss_scale_factor,
                                "actor/kl_coef_bigupdate": self.dense_kl_loss_coef,
                            }
                        ) 
                    
                    if self.dense_kl_loss: 
                        if self.config.disable_policy_gradient: 
                            loss = kl_loss 
                        else: 
                            loss = policy_loss + kl_loss * self.dense_kl_loss_coef 
                    else: 
                        loss = policy_loss 
                    
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = loss * loss_scale_factor
                    else:
                        loss = loss * loss_scale_factor
                    loss.backward()
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_lora_step()
                mini_batch_metrics = {"actor/grad_norm_sparse": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                
        self.actor_optimizer_lora.zero_grad() 
        self.actor_optimizer_full.zero_grad() 
        
        for module in self.actor_module.modules(): 
            if "attention" in module.__class__.__name__.lower(): 
                module.use_full_dense = True 
        
        return metrics 
