SERVER ?= 0.0.0.0
EMBD_PORT ?= 8000
GPU_DEVICE ?= 3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-8B
EMBD_GPU_MEMORY_UTILIZATION ?= 0.9
EMBD_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
EMBD_API_KEY ?= EMPTY
RETRIEVER_JSONL ?= artifacts/summary_retriever_models.jsonl
CHROMA_ROOT ?= artifacts/chroma
ONLY_FIELDS ?= description
FIELD ?= description
TOP_K ?= 10
QUERY ?= I want to read pdf 

MMR_PORT ?= 8001
MMR_GPU_DEVICE ?= 3
MMR_MODEL ?= Qwen/Qwen3-Embedding-8B
MMR_GPU_MEMORY_UTILIZATION ?= 0.5
MMR_BASE_URL ?= http://127.0.0.1:$(MMR_PORT)/v1
MMR_API_KEY ?= EMPTY
MMR_LAMBDA ?= 0.5
MMR_BATCH_SIZE ?= 64
MMR_MAX_CONCURRENCY ?= 32

TRAIN_CONFIG_PATH ?= configs/train.yaml
EVAL_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EVAL_VAL_PARQUET ?= artifacts/val.parquet
EVAL_FORCE_REINDEX ?= false
EVAL_START_VLLM ?= true
EVAL_VLLM_GPU ?= 3
EVAL_VLLM_PORT ?= 8002
EVAL_VLLM_BASE_URL ?= http://127.0.0.1:$(EVAL_VLLM_PORT)/v1
EVAL_VLLM_BATCH_SIZE ?= 512
EVAL_VLLM_MAX_CONCURRENCY ?= 32
EVAL_MAX_VAL_ROWS ?= 0
EVAL_MODE ?= full
EVAL_MODELS ?=
EVAL_WANDB_PROJECT ?= ast-skills-retriever
EVAL_WANDB_ENTITY ?=
EVAL_RUN_NAME ?= validation-parquet-eval
EVAL_USE_HF_ENCODER ?= false
EVAL_DENSE_MODELS ?= Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-4B,Qwen/Qwen3-Embedding-8B,vespa-engine/colbert,sentence-transformers/bert-large-nli-mean-tokens

MINED_INPUT_PARQUET ?= artifacts/retriever_training/train.parquet
MINED_OUTPUT_PARQUET ?= artifacts/retriever_training/training_data.parquet
MINED_CHROMA_ROOT ?= artifacts/chroma_train_builder
MINED_EMBEDDING_MODEL ?= Qwen/Qwen3-Embedding-0.6B
MINED_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
MINED_API_KEY ?= EMPTY
MINED_EMBED_BATCH_SIZE ?= 256
MINED_MAX_CONCURRENCY ?= 32
MINED_RETRIEVAL_TOP_K ?= 37
MINED_DROP_TOP_K ?= 5
MINED_KEEP_NEGATIVES ?= 32
MINED_RRF_K ?= 60
MINED_INCLUDE_NEGATIVE_DESCRIPTIONS ?= false

.PHONY: vllm-embd-serve
vllm-embd-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(EMBD_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

.PHONY: vllm-mmr-serve
vllm-mmr-serve:
	CUDA_VISIBLE_DEVICES=$(MMR_GPU_DEVICE) uv run vllm serve $(MMR_MODEL) \
		--gpu-memory-utilization $(MMR_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(MMR_PORT) \
		--host $(SERVER)

.PHONY: build-dataset
build-dataset:
	uv run python -m ast_skills.train.build_dataset \
		--base_url $(MMR_BASE_URL) \
		--api_key $(MMR_API_KEY) \
		--embedding_model $(MMR_MODEL) \
		--mmr_lambda $(MMR_LAMBDA) \
		--batch_size $(MMR_BATCH_SIZE) \
		--max_concurrency $(MMR_MAX_CONCURRENCY)

.PHONY: build-retriever-chroma
build-retriever-chroma:
	uv run python -m ast_skills.retriever.chroma_embeddings \
		--input_jsonl_path $(RETRIEVER_JSONL) \
		--output_root_dir $(CHROMA_ROOT) \
		--embedding_base_url $(EMBD_BASE_URL) \
		--embedding_model $(EMBD_MODEL) \
		--api_key $(EMBD_API_KEY) \
		$(if $(ONLY_FIELDS),--only_fields $(ONLY_FIELDS))

.PHONY: run-retriever
run-retriever:
	nohup $(MAKE) vllm-embd-serve >> retriever.log 2>&1 &

.PHONY: streamlit-clusters
streamlit-clusters:
	uv run streamlit run ast_skills/retriever/cluster_visualizer.py

.PHONY: retriever-search-semantic
retriever-search-semantic:
	uv run python -m ast_skills.retriever.search semantic --query "$(QUERY)" --field $(FIELD) --root_dir $(CHROMA_ROOT) --embedding_base_url $(EMBD_BASE_URL) --embedding_model $(EMBD_MODEL) --api_key $(EMBD_API_KEY) --limit $(TOP_K)

.PHONY: retriever-search-sparse
retriever-search-sparse:
	uv run python -m ast_skills.retriever.search sparse --query "$(QUERY)" --field $(FIELD) --root_dir $(CHROMA_ROOT) --limit $(TOP_K)

.PHONY: retriever-search-hybrid
retriever-search-hybrid:
	uv run python -m ast_skills.retriever.search hybrid --query "$(QUERY)" --field $(FIELD) --root_dir $(CHROMA_ROOT) --embedding_base_url $(EMBD_BASE_URL) --embedding_model $(EMBD_MODEL) --api_key $(EMBD_API_KEY) --limit $(TOP_K)

.PHONY: retriever-train
retriever-train:
	uv run python -m ast_skills.retriever.train_sentence_transformer train_from_config \
		--config_path $(TRAIN_CONFIG_PATH)

.PHONY: retriever-evaluate
retriever-evaluate:
	uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_from_config \
		--config_path configs/train.yaml

.PHONY: retriever-evaluate-validation
retriever-evaluate-validation:
	uv run python -m ast_skills.evaluation.evaluate_retriever evaluate_validation_parquet \
		--validation_parquet $(EVAL_VAL_PARQUET) \
		--retrieval_model $(EVAL_MODEL) \
		--force_reindex $(EVAL_FORCE_REINDEX) \
		--start_vllm_server $(EVAL_START_VLLM) \
		--vllm_gpu_device $(EVAL_VLLM_GPU) \
		--vllm_port $(EVAL_VLLM_PORT) \
		--vllm_base_url $(EVAL_VLLM_BASE_URL) \
		--vllm_batch_size $(EVAL_VLLM_BATCH_SIZE) \
		--vllm_max_concurrency $(EVAL_VLLM_MAX_CONCURRENCY) \
		--wandb_project $(EVAL_WANDB_PROJECT) \
		--wandb_entity $(EVAL_WANDB_ENTITY) \
		--run_name $(EVAL_RUN_NAME) \
		--use_hf_encoder $(EVAL_USE_HF_ENCODER) \
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
		"$(EVAL_WANDB_PROJECT)" \
		"$(EVAL_WANDB_ENTITY)"

.PHONY: build-mined-training-data
build-mined-training-data:
	uv run python -m ast_skills.train.generate_training_data \
		--input_parquet $(MINED_INPUT_PARQUET) \
		--output_parquet $(MINED_OUTPUT_PARQUET) \
		--chroma_root_dir $(MINED_CHROMA_ROOT) \
		--embedding_model $(MINED_EMBEDDING_MODEL) \
		--embedding_base_url $(MINED_BASE_URL) \
		--api_key $(MINED_API_KEY) \
		--embedding_batch_size $(MINED_EMBED_BATCH_SIZE) \
		--max_concurrency $(MINED_MAX_CONCURRENCY) \
		--retrieval_top_k $(MINED_RETRIEVAL_TOP_K) \
		--drop_top_k $(MINED_DROP_TOP_K) \
		--keep_negatives $(MINED_KEEP_NEGATIVES) \
		--rrf_k $(MINED_RRF_K) \
		--include_negative_descriptions $(MINED_INCLUDE_NEGATIVE_DESCRIPTIONS)
