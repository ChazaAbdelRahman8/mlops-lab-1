from pathlib import Path
import argparse

import mlflow
import mlflow.pytorch as mlflow_pytorch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

NUM_CLASSES = 11
IMAGE_SIZE = 128

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on the Food-11 dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["processed", "mini"],
        default="mini",
        help="Dataset to use: processed or mini",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )

    return parser.parse_args()


# ============================================================
# Dataset loading
# ============================================================

def get_dataset_root(dataset_name):
    if dataset_name == "mini":
        return DATA_DIR / "food11_processed_mini"

    return DATA_DIR / "food11_processed"


def get_dataloaders(dataset_name, batch_size):
    dataset_root = get_dataset_root(dataset_name)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_root}"
        )

    training_path = dataset_root / "training"
    validation_path = dataset_root / "validation"
    evaluation_path = dataset_root / "evaluation"

    for split_path in [
        training_path,
        validation_path,
        evaluation_path,
    ]:
        if not split_path.exists():
            raise FileNotFoundError(
                f"Dataset split not found:\n{split_path}"
            )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------
    # Images were already resized to 128x128 in data.py.
    #
    # We still use ImageNet normalization because ResNet18
    # was pretrained on ImageNet.
    # --------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    # --------------------------------------------------------
    # ImageFolder datasets
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        root=training_path,
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        root=validation_path,
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        root=evaluation_path,
        transform=transform,
    )

    # Ensure all splits use the same classes
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation classes do not match."
        )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Training and evaluation classes do not match."
        )

    print()
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Dataset: {dataset_name}")
    print(f"Dataset path: {dataset_root}")

    print()
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Evaluation images: {len(test_dataset)}")

    print()
    print("Classes:")

    for class_id, class_name in enumerate(
        train_dataset.classes
    ):
        print(
            f"{class_id}: {class_name}"
        )

    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(train_dataset.classes)}."
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
    )


# ============================================================
# Model
# ============================================================

def build_model(num_classes):
    print()
    print("=" * 60)
    print("Model")
    print("=" * 60)

    print("Loading pretrained ResNet18...")

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # ResNet18 originally outputs 1000 ImageNet classes.
    # Replace the final fully connected layer with 11 outputs.

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    print(
        f"Final layer changed to "
        f"{num_classes} output classes."
    )

    return model


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels,
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training dataset contains no images."
        )

    average_loss = (
        running_loss / total_samples
    )

    return average_loss


# ============================================================
# Validation / evaluation
# ============================================================

def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            predictions = outputs.argmax(
                dim=1
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Evaluation dataset contains no images."
        )

    average_loss = (
        running_loss / total_samples
    )

    accuracy = (
        correct / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


# ============================================================
# Main training function
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("=" * 60)
    print("Food-11 ResNet18 Training")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA runtime:",
            torch.version.cuda,
        )
    else:
        print("CUDA is not available.")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        classes,
    ) = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        num_classes=len(classes)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # ========================================================
    # MLflow run
    # ========================================================

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        print()
        print("=" * 60)
        print("MLflow Run")
        print("=" * 60)

        print(f"Run ID: {run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "optimizer": "Adam",
                "num_classes": len(classes),
                "image_size": IMAGE_SIZE,
                "device": str(device),
            }
        )

        # ====================================================
        # Epoch loop
        # ====================================================

        for epoch in range(args.epochs):
            print()
            print(
                f"Epoch "
                f"{epoch + 1}/{args.epochs}"
            )

            print("-" * 60)

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_loss, val_accuracy = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Validation Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: "
                f"{val_accuracy:.4f}"
            )

            # ------------------------------------------------
            # Log metrics to MLflow
            # ------------------------------------------------

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

        # ====================================================
        # Final test/evaluation
        # ====================================================

        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print()
        print("=" * 60)
        print("Final Test Results")
        print("=" * 60)

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        mlflow.log_metric(
            "test_loss",
            test_loss,
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        # ====================================================
        # Log model to MLflow
        # ====================================================

        print()
        print("Logging model to MLflow...")

        # Explicitly use pickle serialization.
        # This avoids MLflow's PT2 format requiring
        # an input_example for graph tracing.

        mlflow_pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        # ====================================================
        # Done
        # ====================================================

        print()
        print("=" * 60)
        print("Training completed successfully")
        print("=" * 60)

        print(f"MLflow Run ID: {run_id}")

        print(
            "Open MLflow at:"
        )

        print(TRACKING_URI)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()