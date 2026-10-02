# Local runs use uv. Docker is an optional single-GPU path.
IMAGE ?= qwen-tuning

train_local:
	uv run src/finetune.py --config configs/base.yaml

test_local:
	uv run src/zero_shot_eval.py --config configs/base.yaml --model tuned

test_base:
	uv run src/zero_shot_eval.py --config configs/base.yaml --model base

demo_model:
	uv run src/finetune.py --config configs/test.yaml

demo_base:
	uv run src/zero_shot_eval.py --config configs/test.yaml

demo_tuned:
	uv run src/zero_shot_eval.py --config configs/test.yaml --model tuned

show_mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# ----- Container config -----
PROJECT_DIR := $(shell pwd)
DATA_DIR ?= $(PROJECT_DIR)/data
RESULTS_DIR ?= $(PROJECT_DIR)/results
MLRUNS_DIR ?= $(PROJECT_DIR)/mlruns
HF_CACHE ?= $(PROJECT_DIR)/.hf-cache
PORT ?= 5000

CTR_APP := /app
CTR_DATA := /app/data
CTR_OUT := /app/results
CTR_MLRUNS := /app/mlruns
CTR_HF := /cache/hf

RUN_SYS := --ipc=host --shm-size=8g
RUN_VOL := -v $(DATA_DIR):$(CTR_DATA) \
           -v $(RESULTS_DIR):$(CTR_OUT) \
           -v $(MLRUNS_DIR):$(CTR_MLRUNS) \
           -v $(HF_CACHE):$(CTR_HF)
RUN_ENV_ONLINE := -e HF_HOME=$(CTR_HF) -e TRANSFORMERS_CACHE=$(CTR_HF)
RUN_ENV_OFFLINE := $(RUN_ENV_ONLINE) -e TRANSFORMERS_OFFLINE=1
RUN_ENV := $(RUN_ENV_ONLINE)

ifeq ($(OFFLINE),1)
RUN_ENV := $(RUN_ENV_OFFLINE)
endif

.PHONY: build prepare train eval mlflow_ui shell clean train_local test_local test_base demo_model demo_base demo_tuned show_mlflow

build:
	docker build -t $(IMAGE) .

prepare:
	mkdir -p $(DATA_DIR) $(RESULTS_DIR) $(MLRUNS_DIR) $(HF_CACHE)

train: prepare
	docker run --rm -it --gpus all $(RUN_SYS) $(RUN_ENV) $(RUN_VOL) \
	  -e CONFIG=configs/base.yaml \
	  $(IMAGE) finetune

eval: prepare
	docker run --rm --gpus all $(RUN_SYS) $(RUN_ENV) $(RUN_VOL) \
	  -e CONFIG=configs/base.yaml \
	  $(IMAGE) eval --model tuned

mlflow_ui: prepare
	docker run --rm -p $(PORT):5000 \
	  -v $(MLRUNS_DIR):$(CTR_MLRUNS) \
	  $(IMAGE) python -m mlflow ui \
	  --backend-store-uri $(CTR_MLRUNS) \
	  --host 0.0.0.0 --port 5000

shell: prepare
	docker run --rm -it --gpus all $(RUN_SYS) $(RUN_ENV) $(RUN_VOL) \
	  -w $(CTR_APP) $(IMAGE) bash

clean:
	rm -rf $(RESULTS_DIR)/* $(MLRUNS_DIR)/*
