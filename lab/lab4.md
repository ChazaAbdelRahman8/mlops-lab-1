# Lab 4 - Orchestrating the App with Docker Compose

In this lab, the MLflow tracking server, inference API, and Streamlit frontend are orchestrated together using Docker Compose.

The three services are:

- `mlflow`: MLflow tracking server and Model Registry
- `inference`: FastAPI service that loads the registered Food-11 model
- `frontend`: Streamlit application used to upload an image and display the prediction

The services communicate through the private network automatically created by Docker Compose.

---

## Question 1

If no volume is mounted at `/mlflow-data`, the MLflow database and model artifacts are stored inside the writable filesystem of the MLflow container.

Stopping the container does not immediately remove this data because the container still exists.

However, once the container is removed, its writable filesystem is deleted. Starting another container from the same image creates a fresh filesystem, so the MLflow UI will be empty again.

Therefore, persistent storage is needed if we want experiments, registered models, aliases, and artifacts to survive container removal.

---

## Question 2

A named volume is useful because Docker manages its location and lifecycle independently from the containers.

This allows the MLflow database and artifacts to remain available even if the MLflow container is removed and recreated.

A bind mount would also work. For example, a directory on the host could be mounted to `/mlflow-data`.

However, a bind mount depends on a specific host filesystem path, while a named volume is managed directly by Docker and makes the Compose configuration more portable between machines.

---

## Question 3

In Lab 3, MLflow was running directly on the host machine. Therefore, the inference container could not reach it using `127.0.0.1`, because `127.0.0.1` inside the container refers to the container itself.

We used `host.docker.internal` to reach the host machine.

In Lab 4, both `mlflow` and `inference` are Docker Compose services connected to the same Compose network.

Docker Compose provides internal DNS that resolves service names automatically. Therefore:

`http://mlflow:5000`

resolves to the MLflow container from inside the inference container.

---

## Question 4

The frontend reads `INFERENCE_URL` from an environment variable so that the same frontend image can be used in different environments.

Inside Docker Compose, the value is:

`http://inference:8000`

because `inference` is the service name.

If the frontend image is run independently outside Compose, the environment variable can instead point to another address, such as:

`http://127.0.0.1:8000`

This avoids hardcoding a Docker Compose-specific hostname in the application.

---

## Question 5

The inference service does not publish port `8000` to the host because it only needs to communicate with the frontend service.

Both services are connected to the same Docker Compose network, so the frontend can reach the inference API directly using:

`http://inference:8000`

The MLflow and frontend services publish ports because they need to be accessed directly by the user through the host machine.

The published ports are:

- MLflow: `5000`
- Frontend: `8501`

---

## Question 6

`depends_on` controls the order in which containers are started, but it does not guarantee that the application inside a container is ready.

Therefore, the MLflow container may have started while the MLflow server is still initializing.

If the inference service immediately executes:

`mlflow.pyfunc.load_model(...)`

and MLflow is not ready yet, the model loading request can fail.

Since the model is loaded during FastAPI startup, an unhandled connection error causes application startup to fail and the inference container exits.

This can be observed using:

```bash
docker compose logs inference
```

A more robust production solution would use a health check and/or retry logic before loading the model.

---

## Question 7

Running:

```bash
docker compose ps
```

shows that the `mlflow` and `frontend` services have ports published to the host.

The expected published ports are:

- `mlflow`: `5000:5000`
- `frontend`: `8501:8501`

The `inference` service does not have a published host port.

This matches the `docker-compose.yml` configuration because the frontend communicates with inference through the private Compose network using:

`http://inference:8000`

Therefore, inference does not need to expose port `8000` directly to the host.

---

## Question 8

The inference service loads the model only once when the application starts.

Therefore, if the `champion` alias is moved to a newer registered model version while the inference container is already running, the running application continues using the old model that is already loaded in memory.

Refreshing the frontend does not reload the model.

To load the new model version, the inference service can be restarted with:

```bash
docker compose restart inference
```

When the container starts again, `serve.py` executes:

`mlflow.pyfunc.load_model("models:/food11@champion")`

again and resolves the `champion` alias to its current model version.

---

## Question 9

A rebuild is not required because the model itself is not baked into the inference Docker image.

The Docker image contains the application code, Python environment, and dependencies required to run the inference API.

The registered model is fetched from MLflow when the inference container starts.

Therefore:

```bash
docker compose restart inference
```

is enough to make the application resolve the `champion` alias again and load the current model version.

This separates application deployment from model deployment.

---

## Question 10

Running:

```bash
docker compose down
docker compose up
```

removes and recreates the containers, but the named volume remains.

Therefore, the MLflow database, registered models, aliases, and model artifacts remain available after the stack is started again.

The named volume is declared as:

```yaml
volumes:
  mlflow-data:
```

and mounted at:

```text
/mlflow-data
```

However, running:

```bash
docker compose down -v
```

also deletes the named volume.

After:

```bash
docker compose up
```

MLflow starts with a new empty volume. The previous experiments, registered models, aliases, and artifacts are no longer available because their persistent storage was deleted.

---

## Question 11

Docker Compose is designed mainly for orchestrating containers on a single machine.

To run multiple replicas of the inference service behind a load balancer and provide high availability across several machines, a container orchestration platform such as Kubernetes would be needed.

Such a platform can provide features including:

- multiple replicas of a service
- service discovery
- load balancing
- automatic container restart
- health checks
- rolling updates
- scheduling containers across multiple machines
- scaling services

For MLflow to survive a machine failure, its state should also not depend on storage located on only one machine.

For example, MLflow could use:

- an external database such as PostgreSQL for backend metadata
- shared or object storage such as S3 for model artifacts

This would allow the MLflow service itself to be recreated on another machine without losing its persistent state.

---

## Docker Compose Architecture

The resulting architecture is:

```text
                    Host machine
                         |
          +--------------+--------------+
          |                             |
     localhost:5000                localhost:8501
          |                             |
          v                             v
     +---------+                  +-----------+
     | MLflow  |                  | Frontend  |
     | :5000   |                  | :8501     |
     +---------+                  +-----------+
          ^                             |
          |                             |
          | http://mlflow:5000          | http://inference:8000
          |                             v
          |                       +-----------+
          +-----------------------| Inference |
                                  | :8000     |
                                  +-----------+

MLflow data
     |
     v
mlflow-data named volume
```

Docker Compose automatically creates a private network for the services.

The service names `mlflow`, `inference`, and `frontend` act as DNS hostnames inside this network.

---

## Main Commands Used

Validate the Compose file:

```bash
docker compose config
```

Build and start the complete stack:

```bash
docker compose up --build
```

Check running services:

```bash
docker compose ps
```

Inspect inference logs:

```bash
docker compose logs inference
```

Restart only the inference service:

```bash
docker compose restart inference
```

Stop the stack while preserving the named volume:

```bash
docker compose down
```

Stop the stack and delete the named volume:

```bash
docker compose down -v
```

---

## Commit

The Lab 4 files are committed with:

```bash
git add mlflow/Dockerfile frontend/Dockerfile frontend/app.py docker-compose.yml
git commit -m "Orchestrate mlflow, inference and frontend with Docker Compose"
git push
```