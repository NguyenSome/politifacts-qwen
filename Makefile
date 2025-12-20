# ----- Local env tasks -----
# Load .env file if it exists (convert to Makefile variables)
ifneq (,$(wildcard .env))
    $(foreach line,$(shell grep -v '^#' .env | grep -v '^$$'),$(eval $(line)))
endif

REGION ?= us-west-2
IMAGE ?= qwen-tuning
ACCOUNT_ID ?= $(AWS_ACCOUNT_ID)

train_local:
		uv run src/finetune.py --config configs/base.yaml

test_local: 
		uv run src/finetune_eval.py

test_base:
		uv run src/zero_shot_eval.py

demo_model:
		uv run src/finetune.py --config configs/test.yaml

demo_base: 
		uv run src/zero_shot_eval.py --config configs/test.yaml

show_mlflow:
		uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

aws_spot:
		aws ec2 request-spot-instances \
		--region $(REGION) \
		--instance-count 1 \
		--type persistent \
		--launch-specification file://spot-spec.json

ecr_login:
		aws ecr get-login-password --region $(REGION) \
		| docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com

build_and_push:
		docker build -t $(IMAGE) .
		docker tag $(IMAGE):latest $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/qwen-tuning:latest
		docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE):latest

attach:
		aws ec2 attach-volume --volume-id ${VOLUME_ID} --instance-id ${INSTANCE_ID} --device /dev/sdf --region $(REGION)

mount:
		sudo mkdir -p /mnt/ebs && sudo mount /dev/nvme2n1 /mnt/ebs

ssh_login:
		aws ecr get-login-password --region us-west-2 \
		| docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com
		
# ----- Container Configs -----
PROJECT_DIR := $(shell pwd)
DATA_DIR ?= $(PROJECT_DIR)/data
RESULTS_DIR ?= $(PROJECT_DIR)/results
MLRUNS_DIR ?= $(PROJECT_DIR)/mlruns
HF_CACHE ?= $(PROJECT_DIR)/.hf-cache
PORT ?= 5000

# Container paths
CTR_APP     := /app
CTR_DATA    := /app/data
CTR_OUT     := /app/results
CTR_MLRUNS  := /app/mlruns
CTR_HF      := /cache/hf

RUN_SYS := --ipc=host --shm-size=8g
RUN_VOL := -v $(DATA_DIR):$(CTR_DATA) \
           -v $(RESULTS_DIR):$(CTR_OUT) \
           -v $(MLRUNS_DIR):$(CTR_MLRUNS) \
           -v $(HF_CACHE):$(CTR_HF)
RUN_ENV_ONLINE  := -e HF_HOME=$(CTR_HF) -e TRANSFORMERS_CACHE=$(CTR_HF)
RUN_ENV_OFFLINE := $(RUN_ENV_ONLINE) -e TRANSFORMERS_OFFLINE=1
RUN_ENV := $(RUN_ENV_ONLINE)

ifeq ($(OFFLINE),1)
RUN_ENV := $(RUN_ENV_OFFLINE)
endif

# -------- Container tasks --------
.PHONY: build prepare train eval mlflow_ui shell clean

build:
	docker build -t $(IMAGE) .

prepare:
	mkdir -p $(DATA_DIR) $(RESULTS_DIR) $(MLRUNS_DIR) $(HF_CACHE)

train: prepare
	docker run --rm --gpus all -it --rm $(RUN_SYS) $(RUN_ENV) $(RUN_VOL) \
	  $(IMAGE) finetune --config configs/base.yaml

eval: prepare
	docker run --rm --gpus all $(RUN_SYS) $(RUN_ENV) $(RUN_VOL) \
	  $(IMAGE) eval --config configs/test.yaml

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
	rm -rf $(RESULTS_DIR)/* $(MLRUNS_DIR)/*  # keep HF cache between runs
