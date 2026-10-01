## Question 1

The registered `food11` model was assigned **Version 1**.

A logged model artifact belongs to a specific MLflow training run and represents the model produced by that run.

A registered model is stored in the MLflow Model Registry under a shared name, such as `food11`, and receives its own version number. This makes it easier to manage, compare, and serve different model versions independently from the original training runs.

## Question 2

MLflow replaced the old fixed stages such as `Staging` and `Production` with flexible aliases such as `champion` and `challenger`.

The model is versioned separately from the run that produced it so that different models from different training runs can be managed under the same registered model name.

An alias is more flexible because it is a movable pointer. For example, `champion` currently points to **Version 1**, but later it can be reassigned to a newer version without changing the serving code that loads:

```text
models:/food11@champion

## Question 3

Loading the model with:

```python
mlflow.pyfunc.load_model("models:/food11@champion")
is better than loading a .pth file directly because the application does not need to know the exact model file path or version.

The champion alias points to the model version that should currently be served.

To serve a newer model version, I only need to move the champion alias to the newer registered model version in MLflow. The serving code can stay unchanged

## Question 4

`pyproject.toml` and `uv.lock` are copied first so Docker can cache the dependency installation.

If only `serve.py` changes, Docker reuses the cached dependencies and only rebuilds the source-code layer, which makes the build much faster.

## Question 5

The multi-stage image is smaller than a naive single-stage image because the final image does not include unnecessary build tools and temporary files.

Using:

```bash
docker history food11-api:latest

## Question 6

Without `.dockerignore`, Docker sends unnecessary files such as the dataset, `.venv`, Git history, and MLflow files to the build context. This makes builds slower and can increase the image size.

The most problematic folders are `.venv/` and `data/` because they can be very large and may unnecessarily bloat the build context.
## Question 7

The container cannot use `127.0.0.1:5000` to reach MLflow because `127.0.0.1` inside the container refers to the container itself, not the host machine.

On Docker Desktop for Windows, `host.docker.internal` resolves to the host computer, so the container can use it to access the MLflow server running on the host.
## Question 8

Yes. A new container created from the same Docker image loads the model correctly without rebuilding the image.

The application code and Python dependencies are baked into the Docker image, while the `champion` model is fetched from MLflow at runtime.

This means the same Docker image can serve a different model version if the `champion` alias is reassigned in MLflow.

## Question 9

The Dockerfile is versioned in Git, but the built Docker image currently exists only on the local machine.

To allow another machine, CI runner, or Kubernetes cluster to reliably run the exact same image, the image must be pushed to a container registry such as Docker Hub, GitHub Container Registry, or another registry.

It is also better to use a specific version tag or image digest instead of only `latest`, so the exact image version can be pulled and reproduced.