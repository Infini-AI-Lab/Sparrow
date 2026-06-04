export PYTHONPATH=${pwd}/vortex_torch:$PYTHONPATH 
export src_dir=${pwd} # TamedSparsity root dir 

set -x 
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 
export RAY_TASK_ERROR_VERBOSE=1 

export TMPDIR=/home/xun/yangzho6/tmp 
mkdir -p $TMPDIR 
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions_${USER:-user}_${RANDOM}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" 

export TORCH_COMPILE_DISABLE=1 

project_name=multiple_models_gridsweeptest 
experiment_name=multiple_models_first_nocudagraph 
TP_SIZE=1 
NNODES=1 
N_GPUS_PER_NODE=8 

localdirr=/${src_dir}/examples/amyaml_legacy/data 
logpath=/${src_dir}/ckpts 
# logpath=/workspace/ckpts 


math_train_path=$localdirr/math/train.parquet
deepscaler_preview_train_path=$localdirr/deepscaler_preview/train.parquet 
polaris_combined_train_path=$localdirr/polaris_53k/train.parquet 

gsminfinitop2to24_train_path=$localdirr/gsminfinitop2to24/train.parquet 

aime2024_test_path=$localdirr/aime2024/test.parquet
aime2025_test_path=$localdirr/aime2025/test.parquet 
aime2024x4_test_path=$localdirr/aime2024x4/test.parquet
aime2025x4_test_path=$localdirr/aime2025x4/test.parquet 
aime2026x4_test_path=$localdirr/aime2026x4/test.parquet 
math500_test_path=$localdirr/math500/test.parquet 
gsm8k_test_path=$localdirr/gsm8k/test.parquet 
amc_test_path=$localdirr/amc/test.parquet
amc23_test_path=$localdirr/amc23/test.parquet
amc24_test_path=$localdirr/amc24/test.parquet 

gsminfinitop2to24_test_path=$localdirr/gsminfinitop2to24/test.parquet 

train_files="['$polaris_combined_train_path']" 
test_files="['$aime2025x4_test_path']" 

mkdir -p data-log/$project_name 

unset ROCR_VISIBLE_DEVICES 
unset HIP_VISIBLE_DEVICES 

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.max_prompt_length=3000 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_double_min_sample=False \
    actor_rollout_ref.actor.double_min_use_latest_logits=False \
    actor_rollout_ref.actor.double_min_apply_trust_region=False \
    actor_rollout_ref.actor.tis_imp_ratio_cap=-1 \
    actor_rollout_ref.actor.log_probs_to_keep=20 \
    actor_rollout_ref.actor.double_min_upclip_ratio=3.0 \
    actor_rollout_ref.actor.generalized_double_min_acceptance_threshold=1.0 \
    actor_rollout_ref.actor.double_min_use_legacy_style_loss=True \
    actor_rollout_ref.rollout.cuda_graph_turnon=True \
    actor_rollout_ref.rollout.sparse_rollout=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend='flashinfer' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_module_name='block_sparse_attention' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_vortex_sparsity=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_bos=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_block_reserved_eos=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_layers_skip=[0,1] \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_schedule_policy='qwen3-4b-0.92' \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_probs_to_keep=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_dense_only=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="${logpath}/$project_name/$experiment_name" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.save_freq=10 \
    trainer.test_freq=100 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_epochs=2000 "$@" 2>&1 
