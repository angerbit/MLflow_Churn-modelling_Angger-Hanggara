# MLProject for Churn Model

This MLProject contains:

- `modelling.py` — training script with manual MLflow logging
- `conda.yaml` — conda environment for MLflow and model dependencies
- `MLproject` — MLflow project entry point
- `namadataset_preprocessing/` — dataset folder placeholder
- `DOCKER_HUB.txt` — Docker Hub image reference placeholder

## Local run

```bash
cd MLProject
python modelling.py --dataset-path namadataset_preprocessing --test-size 0.2 --random-state 42 --max-iter 1000 --experiment-name churn_modeling
```

## MLflow project run

```bash
cd MLProject
mlflow run . -e main \
  -P dataset_path=namadataset_preprocessing \
  -P test_size=0.2 \
  -P random_state=42 \
  -P max_iter=1000
```

## Docker build

In CI we build a Docker image using:

```bash
python -m mlflow build-docker -m . -n <docker-image-name>
```

Replace `<docker-image-name>` with the value from `DOCKER_HUB.txt`.
