SERVER ?= 0.0.0.0
EMBD_PORT ?= 8000
GPU_DEVICE ?= 3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EMBD_GPU_MEMORY_UTILIZATION ?= 0.85
EMBD_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
EMBD_API_KEY ?= EMPTY
RETRIEVER_JSONL ?= artifacts/retriever_models.jsonl
CHROMA_ROOT ?= artifacts/chroma
FIELD ?= description
TOP_K ?= 10
QUERY ?= example query

TRAIN_DATASET_JSONL ?= artifacts/retriever_models_summary.jsonl
TRAIN_OUTPUT_DIR ?= artifacts/sentence_transformers/qwen3-summary
TRAIN_BASE_MODEL ?= Qwen/Qwen3-Embedding-0.6B
TRAIN_EPOCHS ?= 3
TRAIN_BATCH_SIZE ?= 32
TRAIN_LEARNING_RATE ?= 2e-5
TRAIN_WARMUP_STEPS ?= 200
TRAIN_HARD_NEGATIVE_POOL_SIZE ?= 64
TRAIN_DENSE_WEIGHT ?= 1.0
TRAIN_SPARSE_WEIGHT ?= 1.0
TRAIN_MINING_BACKEND ?= bi_encoder
TRAIN_EVAL_BACKEND ?= bi_encoder
TRAIN_QUERY_INSTRUCTION ?=
TRAIN_EVAL_INSTRUCTION ?=
WANDB_PROJECT ?= ast-skills-retriever
WANDB_ENTITY ?=
WANDB_RUN_NAME ?= qwen3-summary-train

EVAL_BACKEND ?= bi_encoder
EVAL_MODEL ?= $(TRAIN_OUTPUT_DIR)
EVAL_QUERY_INSTRUCTION ?=
EVAL_DOCUMENT_INSTRUCTION ?=
EVAL_RUN_NAME ?= retriever-baseline-eval
EVAL_VLLM_BATCH_SIZE ?= 64

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
		--api_key $(EMBD_API_KEY)

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
	uv run python -m ast_skills.retriever.train_sentence_transformer train \
		--dataset_jsonl $(TRAIN_DATASET_JSONL) \
		--output_dir $(TRAIN_OUTPUT_DIR) \
		--base_model_name $(TRAIN_BASE_MODEL) \
		--epochs $(TRAIN_EPOCHS) \
		--batch_size $(TRAIN_BATCH_SIZE) \
		--learning_rate $(TRAIN_LEARNING_RATE) \
		--warmup_steps $(TRAIN_WARMUP_STEPS) \
		--hard_negative_pool_size $(TRAIN_HARD_NEGATIVE_POOL_SIZE) \
		--dense_weight $(TRAIN_DENSE_WEIGHT) \
		--sparse_weight $(TRAIN_SPARSE_WEIGHT) \
		--mining_backend $(TRAIN_MINING_BACKEND) \
		--eval_backend $(TRAIN_EVAL_BACKEND) \
		--query_instruction "$(TRAIN_QUERY_INSTRUCTION)" \
		--eval_instruction "$(TRAIN_EVAL_INSTRUCTION)" \
		--wandb_project $(WANDB_PROJECT) \
		--wandb_entity "$(WANDB_ENTITY)" \
		--run_name $(WANDB_RUN_NAME)

.PHONY: retriever-evaluate
retriever-evaluate:
	uv run python -m ast_skills.retriever.evaluate_retriever evaluate \
		--dataset_jsonl $(TRAIN_DATASET_JSONL) \
		--retrieval_backend $(EVAL_BACKEND) \
		--retrieval_model $(EVAL_MODEL) \
		--query_instruction "$(EVAL_QUERY_INSTRUCTION)" \
		--document_instruction "$(EVAL_DOCUMENT_INSTRUCTION)" \
		--wandb_project $(WANDB_PROJECT) \
		--wandb_entity "$(WANDB_ENTITY)" \
		--run_name $(EVAL_RUN_NAME) \
		--vllm_base_url $(EMBD_BASE_URL) \
		--vllm_api_key $(EMBD_API_KEY) \
		--vllm_batch_size $(EVAL_VLLM_BATCH_SIZE)
