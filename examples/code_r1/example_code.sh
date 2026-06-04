#!/usr/bin/env bash
# Code-R1-style GRPO using this repo's SGLang rollout, keeping the same
# runtime/rollout argument structure as examples/grpo_trainer/multiple_policiymodels_test.sh.
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR/vortex_torch:$PYTHONPATH"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RAY_TASK_ERROR_VERBOSE=1

export TMPDIR="${TMPDIR:-/home/xun/yangzho6/tmp}"
mkdir -p "$TMPDIR"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions_${USER:-user}_${RANDOM}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

export TORCH_COMPILE_DISABLE=1
export REWARD_VALIDATE_WORKERS=32 
ulimit -n 65535

project_name="${PROJECT_NAME:-code_r1_sglang}"
DATASET="${DATASET:-code-r1-21k-taco-codecontests}"
experiment_name="${EXPERIMENT_NAME:-${DATASET}-sglang-grpo}"

TP_SIZE="${TP_SIZE:-1}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE=8 

localdirr="/home/xun/yangzho6/distilldynamcode/examples/amyaml_legacy/data" 
logpath="$PROJECT_DIR/models"

code_train_path="$localdirr/code-r1-21k-taco-codecontests/train.parquet"
lcb_test_path="$localdirr/r1_livecodebench/test.parquet"
humanevalplus_test_path="$localdirr/r1_humanevalplus/test.parquet"
mbppplus_test_path="$localdirr/r1_mbppplus/test.parquet"

train_files="$code_train_path" 
test_files="['$lcb_test_path', '$humanevalplus_test_path', '$mbppplus_test_path']" 

mkdir -p "data-log/$project_name"
mkdir -p "$logpath/$project_name/$experiment_name" 

unset ROCR_VISIBLE_DEVICES 
unset HIP_VISIBLE_DEVICES 

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=16000 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-5e-7}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU:-4}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=19000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_sparse_training=False \
    actor_rollout_ref.actor.bsa_kwargs_nsa_block_size=16 \
    actor_rollout_ref.actor.bsa_kwargs_nsa_block_counts=32 \
    actor_rollout_ref.actor.bsa_kwargs_window_offset=0 \
    actor_rollout_ref.ref.use_sparse_training=False \
    actor_rollout_ref.ref.bsa_kwargs_nsa_block_size=16 \
    actor_rollout_ref.ref.bsa_kwargs_nsa_block_counts=32 \
    actor_rollout_ref.ref.bsa_kwargs_window_offset=0 \
    actor_rollout_ref.rollout.enable_fast_toplogprobs_path=True \
    actor_rollout_ref.rollout.exception_save_steps=\'2,3\' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=19000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=19000 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_double_min_sample=False \
    actor_rollout_ref.actor.double_min_use_latest_logits=False \
    actor_rollout_ref.actor.double_min_apply_trust_region=False \
    actor_rollout_ref.actor.tis_imp_ratio_cap=3.0 \
    actor_rollout_ref.actor.log_probs_to_keep=20 \
    actor_rollout_ref.actor.double_min_upclip_ratio=3.0 \
    actor_rollout_ref.actor.generalized_double_min_acceptance_threshold=1.0 \
    actor_rollout_ref.actor.double_min_use_legacy_style_loss=True \
    actor_rollout_ref.rollout.cuda_graph_turnon=True \
    actor_rollout_ref.rollout.sparse_rollout=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend='flashinfer' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_topk_val=45 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=16 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_module_name='block_sparse_attention' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_vortex_sparsity=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_bos=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_eos=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_layers_skip=[0,1] \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_schedule_policy="qwen3-4b-0.86" \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_probs_to_keep=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.parallel_validate_workers="${REWARD_VALIDATE_WORKERS:-16}" \
    custom_reward_function.path="$PROJECT_DIR/recipe/r1/reward_score.py" \
    custom_reward_function.name=reward_func \
    trainer.check_mode=soft \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="${logpath}/$project_name/$experiment_name" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_epochs="${TOTAL_EPOCHS:-2000}" "$@" 2>&1 
