SERVER ?= 0.0.0.0
EMBD_PORT ?= 8000
GPU_DEVICE ?= 3
GPU_IDS ?= 3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-8B
EMBD_GPU_MEMORY_UTILIZATION ?= 0.9
EMBD_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
EMBD_API_KEY ?= EMPTY
CHROMA_ROOT ?= artifacts/chroma
ONLY_FIELDS ?= summary,description

TRAINING_DATASET_PARQUET ?= artifacts/train.parquet
RETRIEVER_PARQUET ?= $(TRAINING_DATASET_PARQUET)
MINED_TRAIN_PARQUET ?= artifacts/retriever_training/train.parquet
MINED_VALIDATION_PARQUET ?= artifacts/retriever_training/validation.parquet
MINED_RETRIEVAL_POOL_SIZE ?= 200
MINED_WINDOW_START ?= 6
MINED_NEGATIVES_PER_ROW ?= 32
MINED_RANDOM_SEED ?= 13
MINED_VALIDATION_RATIO ?= 0.1
MINED_DATASET_MAX_CONCURRENCY ?= 16

TRAIN_CONFIG_PATH ?= configs/train.config.yaml
HF_DATASET_REPO_ID ?= Agent-Skills-Retriever
HF_MODEL_REPO_ID ?= Qwen3-0.6B-Agent-Skills-Retriever
HF_TRAIN_PARQUET ?= artifacts/retriever_training/train.parquet
HF_VALIDATION_PARQUET ?= artifacts/retriever_training/validation.parquet
HF_TEST_PARQUET ?= artifacts/val.parquet
HF_MODEL_PATH ?= artifacts/sentence_transformers/qwen3-mined-negatives/checkpoint-470
HF_TOKEN ?= 
HF_PRIVATE ?= false

TRAIN_DATA_INPUT_PARQUET ?= artifacts/retriever_training/train.parquet
TRAIN_DATA_OUTPUT_PARQUET ?= artifacts/retriever_training/training_data.parquet
TRAIN_DATA_CHROMA_ROOT ?= artifacts/chroma_train_builder
TRAIN_DATA_EMBEDDING_MODEL ?= Qwen/Qwen3-Embedding-8B
TRAIN_DATA_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
TRAIN_DATA_API_KEY ?= EMPTY
TRAIN_DATA_EMBED_BATCH_SIZE ?= 256
TRAIN_DATA_MAX_CONCURRENCY ?= 256
TRAIN_DATA_RETRIEVAL_TOP_K ?= 37
TRAIN_DATA_DROP_TOP_K ?= 5
TRAIN_DATA_KEEP_NEGATIVES ?= 32
TRAIN_DATA_RRF_K ?= 60
TRAIN_DATA_INCLUDE_NEGATIVE_DESCRIPTIONS ?= false

EVAL_MODEL ?= artifacts/sentence_transformers/qwen3-mined-negatives
EVAL_VAL_PARQUET ?= artifacts/val.parquet
EVAL_WANDB_PROJECT ?= ast-skills-retriever
EVAL_VLLM_PORT ?= 8140
EVAL_VLLM_GPU ?= 1
EVAL_WANDB_ENTITY ?= ad-finance
EVAL_FORCE_REINDEX ?= false
EVAL_START_VLLM ?= true
EVAL_VLLM_BASE_URL ?=
EVAL_VLLM_BATCH_SIZE ?= 512
EVAL_VLLM_MAX_CONCURRENCY ?= 32
EVAL_RUN_NAME ?= validation-parquet-eval
EVAL_USE_HF_ENCODER ?= false
EVAL_MAX_VAL_ROWS ?= 0
# Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-4B,Qwen/Qwen3-Embedding-8B,sentence-transformers/bert-large-nli-mean-tokens

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make build-retriever-chroma            # Build Chroma indexes from a parquet dataset."
	@echo "  make build-mined-negatives-parquet     # Create train/validation parquet with mined negatives."
	@echo "  make retriever-train                   # Train using TRAIN_CONFIG_PATH."
	@echo "  make retriever-train-fsdp-2gpu         # Train with 2-GPU FSDP helper script."
	@echo "  make retriever-evaluate                # Evaluate based on the train config."
	@echo "  make retriever-evaluate-validation     # Evaluate retrieval on a validation parquet."
	@echo "  make retriever-evaluate-validation-bm25# BM25-only validation baseline."
	@echo "  make retriever-evaluate-model-sweep    # Sweep multiple dense models for validation."
	@echo "  make retriever-evaluate-two-gpu        # Evaluate sweep across two GPUs."
	@echo "  make build-mined-training-data         # Build hard-negative training data from train parquet."
	@echo "  make hf-upload-dataset                 # Upload train/val/test parquet files to Hugging Face."
	@echo "  make hf-upload-model                   # Upload a local model directory to Hugging Face."
	@echo ""
	@echo "Run with overridden variables, for example:"
	@echo "  make retriever-train TRAIN_CONFIG_PATH=configs/train.config.yaml"

.PHONY: vllm-embd-serve
vllm-embd-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(MINED_EMBEDDING_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

.PHONY: build-retriever-chroma
build-retriever-chroma:
	uv run python -m ast_skills.retriever.chroma_embeddings \
		--input_parquet_path $(RETRIEVER_PARQUET) \
		--output_root_dir $(CHROMA_ROOT) \
		--embedding_base_url $(EMBD_BASE_URL) \
		--embedding_model $(EMBD_MODEL) \
		--api_key $(EMBD_API_KEY) \
		$(if $(ONLY_FIELDS),--only_fields $(ONLY_FIELDS))

.PHONY: build-mined-negatives-parquet
build-mined-negatives-parquet:
	uv run python -m ast_skills.data_gen.retriever_training_dataset build_retriever_training_dataset \
		--input_parquet_path $(TRAINING_DATASET_PARQUET) \
		--output_train_parquet_path $(MINED_TRAIN_PARQUET) \
		--output_validation_parquet_path $(MINED_VALIDATION_PARQUET) \
		--chroma_root_dir $(CHROMA_ROOT) \
		--embedding_base_url $(EMBD_BASE_URL) \
		--embedding_model $(EMBD_MODEL) \
		--api_key $(EMBD_API_KEY) \
		--random_seed $(MINED_RANDOM_SEED) \
		--validation_ratio $(MINED_VALIDATION_RATIO) \
		--max_concurrency $(MINED_DATASET_MAX_CONCURRENCY) \
		--retrieval_pool_size $(MINED_RETRIEVAL_POOL_SIZE) \
		--window_start_rank $(MINED_WINDOW_START) \
		--negatives_per_row $(MINED_NEGATIVES_PER_ROW)

.PHONY: retriever-train
retriever-train:
	uv run python -m ast_skills.train.train_sentence_transformer train_from_config \
		--config_path $(TRAIN_CONFIG_PATH)

.PHONY: retriever-train-fsdp-2gpu
retriever-train-fsdp-2gpu:
	bash scripts/retriever_train_fsdp_2gpu.sh "$(TRAIN_CONFIG_PATH)" "$(GPU_IDS)"

.PHONY: retriever-evaluate
retriever-evaluate:
	uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_from_config \
		--config_path $(TRAIN_CONFIG_PATH)

.PHONY: retriever-evaluate-validation
retriever-evaluate-validation:
	uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_validation_parquet \
		--validation_parquet $(EVAL_VAL_PARQUET) \
		--retrieval_model $(EVAL_MODEL) \
		--force_reindex $(EVAL_FORCE_REINDEX) \
		--start_vllm_server $(EVAL_START_VLLM) \
		--vllm_gpu_device $(EVAL_VLLM_GPU) \
		--vllm_port $(EVAL_VLLM_PORT) \
		$(if $(EVAL_VLLM_BASE_URL),--vllm_base_url $(EVAL_VLLM_BASE_URL)) \
		--vllm_batch_size $(EVAL_VLLM_BATCH_SIZE) \
		--vllm_max_concurrency $(EVAL_VLLM_MAX_CONCURRENCY) \
		--wandb_project $(EVAL_WANDB_PROJECT) \
		--wandb_entity $(EVAL_WANDB_ENTITY) \
		--run_name $(EVAL_RUN_NAME) \
		--max_validation_rows $(EVAL_MAX_VAL_ROWS)

.PHONY: retriever-evaluate-validation-bm25
retriever-evaluate-validation-bm25:
	uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_validation_bm25_parquet \
		--validation_parquet $(EVAL_VAL_PARQUET) \
		--force_reindex $(EVAL_FORCE_REINDEX) \
		--wandb_project $(EVAL_WANDB_PROJECT) \
		--wandb_entity $(EVAL_WANDB_ENTITY) \
		--run_name $(EVAL_RUN_NAME)-bm25 \
		--max_validation_rows $(EVAL_MAX_VAL_ROWS)

.PHONY: smoke-test
smoke-test:
	$(MAKE) retriever-evaluate-validation EVAL_MAX_VAL_ROWS=5

.PHONY: retriever-evaluate-model-sweep
retriever-evaluate-model-sweep:
	uv run python -m ast_skills.evaluation.run_validation_model_sweep run_model_sweep \
		--validation_parquet $(EVAL_VAL_PARQUET) \
		--mode $(EVAL_MODE) \
		--force_reindex $(EVAL_FORCE_REINDEX) \
		--start_vllm_server $(EVAL_START_VLLM) \
		--vllm_gpu_device $(EVAL_VLLM_GPU) \
		--vllm_port $(EVAL_VLLM_PORT) \
		--vllm_base_url $(EVAL_VLLM_BASE_URL) \
		--vllm_batch_size $(EVAL_VLLM_BATCH_SIZE) \
		--vllm_max_concurrency $(EVAL_VLLM_MAX_CONCURRENCY) \
		--models "$(EVAL_MODELS)"

.PHONY: retriever-evaluate-two-gpu
retriever-evaluate-two-gpu:
	bash scripts/run_validation_two_gpu.sh \
		"$(EVAL_DENSE_MODELS)" \
		"$(EVAL_VAL_PARQUET)" \
		"$(EVAL_WANDB_PROJECT)" 

# "$(EVAL_WANDB_ENTITY)"

.PHONY: build-mined-training-data
build-mined-training-data:
	uv run python -m ast_skills.train.generate_training_data \
		--input_parquet $(TRAIN_DATA_INPUT_PARQUET) \
		--output_parquet $(TRAIN_DATA_OUTPUT_PARQUET) \
		--chroma_root_dir $(TRAIN_DATA_CHROMA_ROOT) \
		--embedding_model $(TRAIN_DATA_EMBEDDING_MODEL) \
		--embedding_base_url $(TRAIN_DATA_BASE_URL) \
		--api_key $(TRAIN_DATA_API_KEY) \
		--embedding_batch_size $(TRAIN_DATA_EMBED_BATCH_SIZE) \
		--max_concurrency $(TRAIN_DATA_MAX_CONCURRENCY) \
		--retrieval_top_k $(TRAIN_DATA_RETRIEVAL_TOP_K) \
		--drop_top_k $(TRAIN_DATA_DROP_TOP_K) \
		--keep_negatives $(TRAIN_DATA_KEEP_NEGATIVES) \
		--rrf_k $(TRAIN_DATA_RRF_K) \
		--include_negative_descriptions $(TRAIN_DATA_INCLUDE_NEGATIVE_DESCRIPTIONS)

.PHONY: hf-upload-dataset
hf-upload-dataset:
	uv run python -m ast_skills.common.huggingface_uploader upload_dataset_splits \
		--dataset_repo_id $(HF_DATASET_REPO_ID) \
		--train_parquet_path $(HF_TRAIN_PARQUET) \
		--validation_parquet_path $(HF_VALIDATION_PARQUET) \
		--test_parquet_path $(HF_TEST_PARQUET) \
		$(if $(HF_TOKEN),--token $(HF_TOKEN)) \
		--private $(HF_PRIVATE)

.PHONY: hf-upload-model
hf-upload-model:
	uv run python -m ast_skills.common.huggingface_uploader upload_model \
		--model_repo_id $(HF_MODEL_REPO_ID) \
		--model_path $(HF_MODEL_PATH) \
		$(if $(HF_TOKEN),--token $(HF_TOKEN)) \
		--private $(HF_PRIVATE)
