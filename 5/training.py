from tqdm import tqdm
import torch
from evaluation import evaluate_model
import matplotlib.pyplot as plt

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader, 
    num_epochs,
    criterion,
    optimizer,
    device, 
    save_path=f"./ckpt/model.pt",
    best_of="accuracy"
):
    """
    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path.
    Inputs:
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        criterion: The loss function [nn.Module]
        optimizer: The optimizer [Any]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
        best_of: The metric to use for selecting the best model [str: "loss" or "accuracy"]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """
    # Move the model to the specified device
    model.to(device)

    # Initialize variables to track the best model
    if best_of == "accuracy":
        best_metric = 0.0  # Higher is better
    elif best_of == "loss":
        best_metric = float('inf')  # Lower is better
    else:
        raise ValueError("best_of should be either 'loss' or 'accuracy'")

    best_epoch = 0
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'best_epoch': 0
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate training loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation loss and accuracy
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        test_acc = evaluate_model(model, test_loader, device=device)

        # Logging the metrics
        training_log['train_loss'].append(train_loss)
        training_log['val_loss'].append(val_loss)
        training_log['train_acc'].append(train_acc)
        training_log['val_acc'].append(val_acc)
        training_log['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")
        

        # Check if the current model is the best based on the selected metric
        if best_of == "accuracy" and val_acc > best_metric:
            best_metric = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {best_epoch} with validation accuracy {best_metric*100:.2f}%")
        elif best_of == "loss" and val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {best_epoch} with validation loss {best_metric:.4f}")

    training_log['best_epoch'] = best_epoch
    return training_log

def plot_training_log(training_log):
    epochs = range(1, len(training_log['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_log['train_loss'], label='Training Loss')
    plt.plot(epochs, training_log['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in training_log['train_acc']], label='Training Accuracy')
    plt.plot(epochs, [acc * 100 for acc in training_log['val_acc']], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()