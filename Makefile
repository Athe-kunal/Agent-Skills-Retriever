SERVER ?= 0.0.0.0
EMBD_PORT ?= 8000
GPU_DEVICE ?= 3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-8B
EMBD_GPU_MEMORY_UTILIZATION ?= 0.9
EMBD_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
EMBD_API_KEY ?= EMPTY
RETRIEVER_JSONL ?= artifacts/summary_retriever_models.jsonl
CHROMA_ROOT ?= artifacts/chroma
ONLY_FIELDS ?= summary,description

TRAINING_DATASET_JSONL ?= artifacts/summary_retriever_models.jsonl
MINED_TRAIN_PARQUET ?= artifacts/retriever_training/train.parquet
MINED_VALIDATION_PARQUET ?= artifacts/retriever_training/validation.parquet
MINED_TOP_K ?= 37
MINED_WINDOW_START ?= 6
MINED_WINDOW_END ?= 37
MINED_NEGATIVES_PER_ROW ?= 32
MINED_RANDOM_SEED ?= 13
MINED_VALIDATION_RATIO ?= 0.1
MINED_MAX_CONCURRENCY ?= 16

TRAIN_CONFIG_PATH ?= configs/train.config.yaml

.PHONY: vllm-embd-serve
vllm-embd-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(EMBD_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

.PHONY: build-retriever-chroma
build-retriever-chroma:
	uv run python -m ast_skills.retriever.chroma_embeddings \
		--input_jsonl_path $(RETRIEVER_JSONL) \
		--output_root_dir $(CHROMA_ROOT) \
		--embedding_base_url $(EMBD_BASE_URL) \
		--embedding_model $(EMBD_MODEL) \
		--api_key $(EMBD_API_KEY) \
		$(if $(ONLY_FIELDS),--only_fields $(ONLY_FIELDS))

.PHONY: build-mined-negatives-parquet
build-mined-negatives-parquet:
	uv run python -m ast_skills.data_gen.retriever_training_dataset build_retriever_training_dataset \
		--input_jsonl_path $(TRAINING_DATASET_JSONL) \
		--output_train_parquet_path $(MINED_TRAIN_PARQUET) \
		--output_validation_parquet_path $(MINED_VALIDATION_PARQUET) \
		--chroma_root_dir $(CHROMA_ROOT) \
		--embedding_base_url $(EMBD_BASE_URL) \
		--embedding_model $(EMBD_MODEL) \
		--api_key $(EMBD_API_KEY) \
		--random_seed $(MINED_RANDOM_SEED) \
		--validation_ratio $(MINED_VALIDATION_RATIO) \
		--max_concurrency $(MINED_MAX_CONCURRENCY) \
		--top_k $(MINED_TOP_K) \
		--window_start_rank $(MINED_WINDOW_START) \
		--window_end_rank $(MINED_WINDOW_END) \
		--negatives_per_row $(MINED_NEGATIVES_PER_ROW)

.PHONY: retriever-train
retriever-train:
	uv run python -m ast_skills.train.train_sentence_transformer train_from_config \
		--config_path $(TRAIN_CONFIG_PATH)
