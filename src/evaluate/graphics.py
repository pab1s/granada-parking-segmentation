import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_dfs(csv_paths, dataframe_names, column_name, output_path, n_rows=None, style="whitegrid", line_width=2.5):
    """
    Reads CSV files, plots a line plot for the specified column from these files,
    and saves the plot to a specified path, using only the first n_rows of each DataFrame.
    Each line in the plot is labeled with the provided DataFrame names and is thicker.
    The axis titles are bold.
    
    Parameters:
    - csv_paths (List[str]): A list of paths to the CSV files.
    - dataframe_names (List[str]): A list of names for each DataFrame.
    - column_name (str): The name of the column to plot.
    - output_path (str): The path to save the plot.
    - n_rows (int, optional): Number of rows to consider from each DataFrame.
    - style (str, optional): The style of the seaborn plot.
    - line_width (float, optional): The thickness of the lines in the plot.
    """
    if len(csv_paths) != len(dataframe_names):
        raise ValueError("The number of DataFrame names must match the number of CSV paths.")

    try:
        # Create the output directory if it doesn't exist
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            os.makedirs(output_dir)

        # Set the plot style
        sns.set(style=style)
        plt.figure(figsize=(10, 6))

        # Plot each DataFrame separately
        for path, name in zip(csv_paths, dataframe_names):
            df = pd.read_csv(path).head(n_rows)
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame at {path}.")
            sns.lineplot(data=df, x=df.index, y=column_name, label=name, linewidth=line_width)

        # Customize the plot with bold axis titles
        plt.title(f'Comparison of {column_name}', fontweight='bold')
        plt.xlabel('Index', fontweight='bold')
        plt.ylabel(column_name, fontweight='bold')

        # Save the plot to the specified path
        plt.savefig(output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the plot to avoid memory issues
        plt.close()

if __name__ == '__main__':
    csv_paths = [
        'results/logs/pspnet_metrics.csv',
        'results/logs/deeplab_metrics.csv',
        'results/logs/unet_metrics.csv'
    ]
    dataframe_names = ['PSPNet', 'DeepLabV3+', 'Dynamic UNet']
    column_name = 'train_loss'
    output_path = 'results/figures/train_loss_comparison.png'
    n_rows = 50
    style = "darkgrid"

    plot_dfs(csv_paths, dataframe_names, column_name, output_path, n_rows, style)
