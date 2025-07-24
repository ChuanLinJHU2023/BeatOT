import matplotlib.pyplot as plt
import torch
import numpy as np
from tensorflow.python.keras.engine.training_v1 import Model

from models import *

def visualize_domains(datasets, labels, titles, x_limit=None, y_limit=None, with_model=None, single_plot=False, save_path=None):
    """
    Plots multiple domain feature distributions side by side or on a single plot.
    When with_model is provided, predictions are visualized as a smoothed gradient background,
    with the original data overlaid.

    Parameters:
    - datasets: list of arrays, each of shape (n_samples, n_features)
    - labels: list of arrays, each containing 0 or 1 labels
    - titles: list of strings for each plot title
    - x_limit: tuple (xmin, xmax) for axes limits (optional)
    - y_limit: tuple (ymin, ymax) for axes limits (optional)
    - with_model: PyTorch model for prediction visualization (optional)
    - single_plot: bool, if True overlays all datasets in one plot (default False)
    """
    num_datasets = len(datasets)

    # Prepare grid for model prediction if model is provided
    if with_model is not None and x_limit is not None and y_limit is not None:
        grid_x = np.linspace(x_limit[0], x_limit[1], 200)
        grid_y = np.linspace(y_limit[0], y_limit[1], 200)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        with torch.no_grad():
            preds = with_model(grid_tensor).numpy()
        preds = preds.reshape(xx.shape)

    plt.figure(figsize=(6 * num_datasets if not single_plot else 8, 5))

    if single_plot:
        axes = plt.gca()
        # Draw model prediction background once
        if with_model is not None and x_limit is not None and y_limit is not None:
            axes.contourf(xx, yy, 1 - preds, levels=50, cmap='RdBu', alpha=0.8)

        for i in range(num_datasets):
            pos_mask = labels[i] == 1
            neg_mask = labels[i] == 0

            axes.scatter(
                datasets[i][pos_mask, 0],
                datasets[i][pos_mask, 1],
                color='red',
                label='Positive (1)' if i == 0 else "",
                alpha=0.8,
                s=20,
                edgecolors='k'
            )
            axes.scatter(
                datasets[i][neg_mask, 0],
                datasets[i][neg_mask, 1],
                color='blue',
                label='Negative (0)' if i == 0 else "",
                alpha=0.8,
                s=20,
                edgecolors='k'
            )
            # Only set title and legend once
        plt.title('All Data (Overlayed)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if x_limit is not None:
            plt.xlim(x_limit)
        if y_limit is not None:
            plt.ylim(y_limit)
        plt.legend()

    else:
        # Multiple subplots side by side
        for i in range(num_datasets):
            plt.subplot(1, num_datasets, i + 1)
            if with_model is not None and x_limit is not None and y_limit is not None:
                plt.contourf(xx, yy, 1 - preds, levels=50, cmap='RdBu', alpha=0.8)
            pos_mask = labels[i] == 1
            neg_mask = labels[i] == 0

            plt.scatter(
                datasets[i][pos_mask, 0],
                datasets[i][pos_mask, 1],
                color='red',
                label='Positive (1)',
                alpha=0.8,
                s=20,
                edgecolors='k'
            )
            plt.scatter(
                datasets[i][neg_mask, 0],
                datasets[i][neg_mask, 1],
                color='blue',
                label='Negative (0)',
                alpha=0.8,
                s=20,
                edgecolors='k'
            )
            plt.title(titles[i])
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            if x_limit is not None:
                plt.xlim(x_limit)
            if y_limit is not None:
                plt.ylim(y_limit)
            if with_model is None:
                plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_a_case(project_root=None, case_number=0, checkpoint_file_path=None, visualization_file_path=None):
    if project_root==None:
        project_root="./"
    data = np.load(project_root + f'cases/case{case_number}.npz')
    X_source = data['X_source']
    y_source = data['y_source']
    X_target = data['X_target']
    y_target = data['y_target']
    if not checkpoint_file_path:
        visualize_domains([X_source, X_target], [y_source, y_target],
                          ['Source Domain', "Target Domain"],
                          x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=None, save_path=visualization_file_path)
    else:
        num_hidden_units = 16
        list_of_num_hidden_units = [num_hidden_units]
        model = SimpleClassifier(list_of_num_hidden_units)
        model.load_state_dict(torch.load(checkpoint_file_path))
        model.eval()
        visualize_domains([X_source, X_target], [y_source, y_target],
                          ['Source Domain', "Target Domain"],
                          x_limit=(-2.5, 3.5), y_limit=(-3, 3), with_model=model, save_path=visualization_file_path)

# visualize_a_case(case_number=1, checkpoint_file_path="./checkpoints/20250723_215922.pth", visualization_file_path="./images/MyImage3.png")