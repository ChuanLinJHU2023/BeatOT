
import numpy as np
import torch
from utils import *

def print_hyper_parameters(n_samples=None, hyper_parameter_p=None, hyper_parameter_c=None, learning_rate=None, num_epochs=None, list_of_num_hidden_units=None):
    print_format_string("HYPER-PARAMETER", 30)
    if n_samples is not None:
        print(f"N samples: {n_samples}")
    if hyper_parameter_p is not None:
        print(f"Hyper Parameter p: {hyper_parameter_p}")
    if hyper_parameter_c is not None:
        print(f"Hyper Parameter c: {hyper_parameter_c}")
    if learning_rate is not None:
        print(f"Learning Rate: {learning_rate}")
    if num_epochs is not None:
        print(f"Number of Epochs: {num_epochs}")
    if list_of_num_hidden_units is not None:
        print(f"List of Hidden Units: {list_of_num_hidden_units}")


def evaluate_and_print_for_binary_classification(X, y, model, num_samples_to_print=5):
    """
    Evaluates a PyTorch binary classification model on test data
    and prints detailed metrics and sample predictions.

    Args:
        X (np.ndarray): Test features, shape (num_samples, num_features)
        y (np.ndarray): True labels, shape (num_samples,)
        model (torch.nn.Module): Trained PyTorch model
        num_samples_to_print (int): Number of random samples to display
    """
    print_format_string("EVALUATION", 30)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        probs = model(X_tensor).squeeze()
        preds = (probs >= 0.5).long()
        total_samples = len(y)
        correct = (preds == y_tensor).sum().item()
        accuracy = correct / total_samples
        # Accuracy for positive points (Recall)
        positive_mask = y_tensor == 1
        positive_correct = ((preds == 1) & positive_mask).sum().item()
        positive_total = positive_mask.sum().item()
        positive_accuracy = positive_correct / positive_total if positive_total > 0 else float('nan')
        # Accuracy for negative points
        negative_mask = y_tensor == 0
        negative_correct = ((preds == 0) & negative_mask).sum().item()
        negative_total = negative_mask.sum().item()
        negative_accuracy = negative_correct / negative_total if negative_total > 0 else float('nan')
        # Ratio of predictions as positive
        positive_predictions_ratio = preds.sum().item() / total_samples
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        print(f"Accuracy for positive points (Recall): {positive_accuracy * 100:.2f}%" if not np.isnan(
            positive_accuracy) else "No positive samples in true labels.")
        print(f"Accuracy for negative points: {negative_accuracy * 100:.2f}%" if not np.isnan(
            negative_accuracy) else "No negative samples in true labels.")
        print(f"Proportion of samples predicted as positive: {positive_predictions_ratio * 100:.2f}%")
        # Randomly select samples to print
        indices = np.arange(total_samples)
        if total_samples >= num_samples_to_print:
            sample_indices = np.random.choice(indices, size=num_samples_to_print, replace=False)
        else:
            sample_indices = indices
        for idx in sample_indices:
            print(
                f"Sample {idx}: True label = {y[idx]}, Predicted = {preds[idx].item()}, Probability = {probs[idx].item():.4f}"
            )
        return accuracy, positive_accuracy, negative_accuracy, positive_predictions_ratio