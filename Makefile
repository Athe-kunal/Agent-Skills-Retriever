SERVER ?= 0.0.0.0
EMBD_PORT ?= 8000
GPU_DEVICE ?= 3
EMBD_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EMBD_GPU_MEMORY_UTILIZATION ?= 0.85
EMBD_BASE_URL ?= http://127.0.0.1:$(EMBD_PORT)/v1
EMBD_API_KEY ?= EMPTY
RETRIEVER_JSONL ?= artifacts/summary_retriever_models.jsonl
CHROMA_ROOT ?= artifacts/chroma
ONLY_FIELDS ?= description
FIELD ?= summary
TOP_K ?= 10
QUERY ?= I want to read pdf 

TRAIN_CONFIG_PATH ?= configs/train.yaml
EVAL_CONFIG_PATH ?= configs/train.yaml

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
	uv run python -m ast_skills.retriever.evaluate_retriever evaluate_from_config \
		--config_path $(EVAL_CONFIG_PATH)
