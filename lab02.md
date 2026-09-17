# Lab 2 - Model Training and Experiment Tracking with MLflow

## Question 1

After installing:

```bash
uv add mlflow torch torchvision scikit-learn

pyproject.toml was updated with the new dependencies, while uv.lock stored the exact resolved versions and their dependencies to keep the environment reproducible.

Question 2

--backend-store-uri sqlite:///mlflow.db stores MLflow metadata such as experiments, runs, parameters, metrics, and tags.

--default-artifact-root ./mlruns defines where MLflow stores artifacts such as trained models and other generated files.

Metadata describes the experiment, while artifacts are the actual files produced by the experiment.

Question 3

mlflow.db and mlruns/ should not be tracked by Git because they are generated experiment outputs and may change frequently or become large.

They should not be tracked by DVC because DVC is used for dataset versioning, while MLflow already manages experiment metadata and artifacts.

Git → code
DVC → datasets
MLflow → experiments and models
Question 4

When:

mlflow.set_experiment("food11")

is called and the experiment does not exist, MLflow automatically creates a new experiment called food11.

It then appears in the MLflow UI.

Question 5

mlflow.log_param() is used for fixed values such as:

Learning rate
Batch size
Number of epochs
Model architecture

mlflow.log_metric() is used for values that change during training, such as:

Training loss
Validation loss
Validation accuracy

Metrics use a step because their values are recorded at different epochs. Parameters remain constant, so they do not need a step.

Question 6

The MLflow run contains:

Parameters
Training and validation metrics
Test accuracy
Trained model artifact

Since the server was started with:

--default-artifact-root ./mlruns

the model artifacts are stored locally under:

mlruns/
Question 7

The learning rate with the best validation accuracy was:

Best learning rate: [ADD RESULT]
Best val_accuracy: [ADD RESULT]

A higher learning rate is not always better. A learning rate that is too high can make training unstable, while one that is too low can make training very slow.

Question 8

The parallel coordinates plot compares:

Learning rate
Batch size
Validation accuracy

The results show that model performance depends on the combination of learning rate and batch size.

Observed pattern: [ADD YOUR MLFLOW RESULT]
Question 9

After sorting by validation accuracy, the best run was:

Run ID: [ADD RUN ID]
Learning rate: [ADD LR]
Batch size: [ADD BATCH SIZE]
Validation accuracy: [ADD VAL ACCURACY]

This run ID will be used in the next lab.

Run MLflow
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

MLflow UI:

http://127.0.0.1:5000
Train the Model

Example:

uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32