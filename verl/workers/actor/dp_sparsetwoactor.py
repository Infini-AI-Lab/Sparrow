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

import numpy as np 


__all__ = ["DataParallelSparseTwoActor"] 

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelSparseTwoActor(BasePPOActor): 
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, 
                 config: ActorConfig, 
                 actorsparse_module: nn.Module, 
                 actordense_module: nn.Module, 
                 actorsparse_optimizer: torch.optim.Optimizer, 
                 actordense_optimizer: torch.optim.Optimizer, 
    ): 
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_sparse_module = actorsparse_module 
        self.actor_sparse_optimizer = actorsparse_optimizer 
        self.actor_dense_module = actordense_module 
        self.actor_dense_optimizer = actordense_optimizer 
        # role = "Ref" if actor_optimizer is None else "Actor" 
        role = "Ref" if actorsparse_optimizer is None else "Actor" 

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
        
        assert self.actor_sparse_module is not None and self.actor_dense_module is not None, "actor_sparse_module and actor_dense_module must be provided" 
        assert self.actor_sparse_optimizer is not None and self.actor_dense_optimizer is not None, "actor_sparse_optimizer and actor_dense_optimizer must be provided" 
        self._assert_param_disjoint() 
        self._assert_optimizer_match() 
        
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

    def _assert_optimizer_match(self):
        r_set = {id(p) for g in self.actor_sparse_optimizer.param_groups for p in g['params']}
        u_set = {id(p) for g in self.actor_dense_optimizer.param_groups for p in g['params']}

        # No cross-listing
        assert r_set.isdisjoint(u_set), "Optimizers contain overlapping params."

        # Coverage
        all_rollout = {id(p) for p in self.actor_sparse_module.parameters() if p.requires_grad}
        all_update  = {id(p) for p in self.actor_dense_module.parameters() if p.requires_grad}
        assert all_rollout <= r_set, "Some sparse params are not in sparse optimizer."
        assert all_update  <= u_set, "Some dense params are not in dense optimizer." 
    
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

                output = self.actor_sparse_module(
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
                    if not self.actor_sparse_optimizer is None: 
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
                    if not self.actor_sparse_optimizer is None:
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
                if not self.actor_sparse_optimizer is None:
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
                if not self.actor_sparse_optimizer is None:
                    if self.config.use_double_min_sample:
                        topk_idx = full_topk_idx.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)
                        topk_logp = full_topk_logp.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_sparse_module(
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
                    if not self.actor_sparse_optimizer is None:
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

                output = self.actor_dense_module(
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
                    if not self.actor_dense_optimizer is None: 
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
                    if not self.actor_dense_optimizer is None: 
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
                if not self.actor_dense_optimizer is None:
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
                if not self.actor_dense_optimizer is None:
                    if self.config.use_double_min_sample:
                        topk_idx = full_topk_idx.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)
                        topk_logp = full_topk_logp.squeeze(-1)[:, -response_length - 1 : -1, :]  # (bsz, response_length, topk)

            else:  # not using rmpad and no ulysses sp 
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_dense_module(
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
                    print("self.actor_dense_optimizer {}".format(self.actor_dense_optimizer)) 
                    if not self.actor_dense_optimizer is None:
                        if self.config.use_double_min_sample:
                            topk_idx, topk_logp, _ = logits_to_topk(logits, self.config.log_probs_to_keep)
                    
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, topk_idx, topk_logp 

    def _optimizer_sparse_step(self): 
        assert self.config.grad_clip is not None

        if isinstance(self.actor_sparse_module, FSDP):
            grad_norm = self.actor_sparse_module.clip_grad_norm_(max_norm=self.config.grad_clip) 
        elif isinstance(self.actor_sparse_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_sparse_module.parameters(), max_norm=self.config.grad_clip) 
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_sparse_module.parameters(), max_norm=self.config.grad_clip) 

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_sparse_optimizer.zero_grad() 
        else:
            self.actor_sparse_optimizer.step() 
        return grad_norm 

    def _optimizer_dense_step(self): 
        assert self.config.grad_clip is not None

        if isinstance(self.actor_dense_module, FSDP): 
            grad_norm = self.actor_dense_module.clip_grad_norm_(max_norm=self.config.grad_clip) 
        elif isinstance(self.actor_dense_module, FSDPModule): 
            grad_norm = fsdp2_clip_grad_norm_(self.actor_dense_module.parameters(), max_norm=self.config.grad_clip) 
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_dense_module.parameters(), max_norm=self.config.grad_clip) 

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_dense_optimizer.zero_grad() 
        else:
            self.actor_dense_optimizer.step() 
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
        self.actor_sparse_module.eval()

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
            if not self.actor_sparse_optimizer is None:
                if self.config.use_double_min_sample:
                    if not topk_idx is None:
                        topk_idx_lst.append(topk_idx)
                        topk_logp_lst.append(topk_logp)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if not self.actor_sparse_optimizer is None:
            if self.config.use_double_min_sample:
                if topk_idx_lst:
                    topk_idx = torch.concat(topk_idx_lst, dim=0)
                    topk_logp = torch.concat(topk_logp_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if not self.actor_sparse_optimizer is None:
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
        self.actor_dense_module.eval() 

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
                entropy, log_probs, topk_idx, topk_logp = self._forwarddense_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
                
            log_probs_lst.append(log_probs)
            if not self.actor_dense_optimizer is None:
                if self.config.use_double_min_sample:
                    if not topk_idx is None:
                        topk_idx_lst.append(topk_idx)
                        topk_logp_lst.append(topk_logp)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if not self.actor_dense_optimizer is None:
            if self.config.use_double_min_sample:
                if topk_idx_lst:
                    topk_idx = torch.concat(topk_idx_lst, dim=0)
                    topk_logp = torch.concat(topk_logp_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if not self.actor_dense_optimizer is None:
                if self.config.use_double_min_sample:
                    if (topk_idx_lst):
                        topk_idx = restore_dynamic_batch(topk_idx, batch_idx_list)
                        topk_logp = restore_dynamic_batch(topk_logp, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
        return log_probs, entropys, topk_idx, topk_logp 

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_sparse_module.train() 
        self.actor_dense_module.train() 

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            # "old_log_probs", 
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

                self.actor_sparse_optimizer.zero_grad() 
                self.actor_dense_optimizer.zero_grad() 

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"] 
                    
                    # first update the rollout model weights 
                    ######## 
                    oldsparse_log_prob = model_inputs["oldsparse_log_probs"] 
                    # rollout model is strictly onpolicy, we don't need topk_idx and topk_logp 
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None 
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

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
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

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef 

                    if not self.dense_kl_loss: # if we need the kl for multiple policy models, you need to wait until the update model compute the log_probs 
                        print("######### not bigupdate_kl_loss #########") 
                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor 
                        loss.backward() 
                    
                    metrics_dict = {
                            "actor/pg_rollout_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_rollout_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_rollout_kl": ppo_kl.detach().item(),
                            "actor/pg_rollout_clipfrac_lower": pg_clipfrac_lower.detach().item(),                          
                        } 

                    micro_batch_metrics.update(
                        metrics_dict
                    )
                    append_to_dict(metrics, micro_batch_metrics) 
                    
                    # second update the update model weights 
                    ######## 
                    olddense_log_prob = model_inputs["olddense_log_probs"] 
                    olddense_topk_idx = model_inputs["olddense_topk_idx"] if self.config.use_double_min_sample else None
                    olddense_topk_logp = model_inputs["olddense_topk_logp"] if self.config.use_double_min_sample else None
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 or self.config.use_double_min_sample else None 
                    
                    rollout_log_probs_topk_idx = model_inputs["rollout_log_probs_topk_idx"] if self.config.use_double_min_sample else None
                    rollout_log_probs_topk_logprob = model_inputs["rollout_log_probs_topk_logprob"] if self.config.use_double_min_sample else None 

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
                    entropy_update, log_prob_update, topk_idx_update, topk_logp_update = self._forwarddense_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    ) 

                    if on_policy:
                        olddense_log_prob = log_prob_update.detach() 
                        if self.config.use_double_min_sample and self.config.double_min_use_latest_logits: 
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
                            log_prob = log_prob_update, 
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
                        entropy_loss = agg_loss(loss_mat=entropy_update, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss_update = pg_loss_update - entropy_loss * entropy_coeff 
                    else:
                        policy_loss_update = pg_loss_update 

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob_update, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
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
                    loss_update.backward() 
                    
                    metrics_dict = {
                            "actor/pg_loss_update": pg_loss_update.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac_update": pg_clipfrac_update.detach().item(),
                            "actor/ppo_kl_update": ppo_kl_update.detach().item(),
                            "actor/pg_clipfrac_lower_update": pg_clipfrac_lowerupdate.detach().item(),                          
                        }
                    if self.config.use_double_min_sample:
                        double_min_sample_metrics_dict = { # double min sample is only used in the update model 
                            "actor/sampler_acceptance_rate_update": acceptance_rate_update,
                            "actor/ppo_kl_vllm_update": ppo_kl_vllm_update.detach().item(),
                            "actor/ppo_kl_double_mindense": ppo_kl_double_mindense.detach().item() 
                        } 
                        
                        metrics_dict.update(double_min_sample_metrics_dict) 


                    micro_batch_metrics.update(
                        metrics_dict
                    ) 
                    
                    if self.dense_kl_loss: 
                        print("######### bigupdate_kl_loss #########") 
                        update_log_prob = log_prob_update.detach() 
                        klloss = kl_penalty(
                            logprob = log_prob, ref_logprob = update_log_prob, kl_penalty = self.dense_kl_loss_type, 
                        ) 
                        
                        kl_loss = agg_loss(loss_mat=klloss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode) 
                        micro_batch_metrics.update(
                            {
                                "actor/kl_loss_bigupdate": kl_loss.detach().item() * loss_scale_factor, 
                                "actor/kl_coef_bigupdate": self.dense_kl_loss_coef, 
                            }
                        ) 
                        if data.meta_info["stop_update_model"] and self.config.disable_policy_gradient_update_step: 
                            print("######### only update the kl loss #########", flush = True) 
                            loss = kl_loss 
                        else: 
                            print("######### update the policy gradient and kl loss #########", flush = True) 
                            loss = policy_loss + kl_loss * self.dense_kl_loss_coef 
                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = loss * loss_scale_factor 
                        else:
                            loss = loss * loss_scale_factor 
                        loss.backward() 

                    append_to_dict(metrics, micro_batch_metrics) 

                grad_norm = self._optimizer_sparse_step() 
                mini_batch_metrics = {"actor/grad_norm_sparse": grad_norm.detach().item()} 
                append_to_dict(metrics, mini_batch_metrics) 
                
                '''
                if not data.meta_info["stop_update_model"]: 
                    if data.meta_info["global_step"] > 200 and np.mean(metrics["actor/sampler_acceptance_rate_update"]) < 0.9: 
                        print("*** Stop update model gradnorm step ***", flush = True) 
                        mini_batch_metrics = {"actor/stop_update_model": 1, "actor/grad_norm_update": 0} 
                        append_to_dict(metrics, mini_batch_metrics) 
                    else: 
                        print("*** Update model gradnorm step ***", flush = True) 
                        grad_norm = self._optimizer_dense_step() 
                        mini_batch_metrics = {"actor/grad_norm_update": grad_norm.detach().item(), "actor/stop_update_model": 0} 
                        append_to_dict(metrics, mini_batch_metrics) 
                else: 
                    print("*** Stop update model gradnorm step ***", flush = True) 
                    mini_batch_metrics = {"actor/stop_update_model": 1, "actor/grad_norm_update": 0} 
                    append_to_dict(metrics, mini_batch_metrics) 
                ''' 
                
                print("*** Update model gradnorm step ***", flush = True) 
                grad_norm = self._optimizer_dense_step() 
                mini_batch_metrics = {"actor/grad_norm_update": grad_norm.detach().item(), "actor/stop_update_model": 0} 
                append_to_dict(metrics, mini_batch_metrics) 
        self.actor_sparse_optimizer.zero_grad() 
        self.actor_dense_optimizer.zero_grad() 
        return metrics 
