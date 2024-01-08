import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import Recorder, patch, delegates, subplots

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    """
    Extend the fastai Recorder class to plot evaluation metrics.

    This function plots the evolution of training and validation metrics over the course of training.
    It's designed to work as a method of fastai's Recorder class, which tracks the metrics during training.
    The function automatically determines the layout of the subplots based on the number of metrics.

    Parameters:
    - self (Recorder): An instance of fastai's Recorder, attached to a Learner.
    - nrows (int, optional): Number of rows in the subplot grid. Automatically determined if not provided.
    - ncols (int, optional): Number of columns in the subplot grid. Automatically determined if not provided.
    - figsize (tuple, optional): Size of the figure. Automatically determined if not provided.
    - **kwargs: Additional keyword arguments passed to fastai's subplots function.

    Credits: 
    Developed by Ignacio Oguiza - https://forums.fast.ai/t/plotting-metrics-after-learning/69937
    """
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off()
           for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i ==
                0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

def save_metrics_to_csv(learner, file_path='pspnet_metrics.csv', **kwargs):
    """
    Save the training metrics from a fastai Learner to a CSV file.

    This function extracts the recorded metrics from a fastai Learner and writes them to a CSV file.
    Each row in the CSV file corresponds to an epoch, and it includes all the metrics recorded for that epoch.

    Parameters:
    - learner (Learner): A fastai Learner object from which to extract training metrics.
    - file_path (str, optional): The path where the CSV file will be saved. Defaults to 'pspnet_metrics.csv'.
    - **kwargs: Additional keyword arguments (currently not used but included for future extension).

    Outputs:
    - A CSV file containing the training metrics.
    """
    recorder = learner.recorder
    metrics = np.stack(recorder.values)
    names = recorder.metric_names[1:-1]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['epoch'] + names)

        # Write the metrics for each epoch
        for epoch, metric_values in enumerate(metrics):
            writer.writerow([epoch + 1] + list(metric_values))

    print(f"Metrics saved to {file_path}")