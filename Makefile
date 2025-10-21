install:
		pip install --upgrade pip &&\
			pip install uv && uv pip install --system --group default

train:
		uv run src/finetune.py

test: 
		uv run src/finetune_eval.py

test_base:
		uv run src/zero_shot_eval.py

demo_model:
		uv run src/finetune.py --config configs/test.yaml

demo_base: 
		uv run src/baseline_eval.py --config configs/test.yaml

show_mlflow:
		uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

.PHONY: docker
docker_build:

docker_demo: