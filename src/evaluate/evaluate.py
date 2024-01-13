from fastai.vision.all import *
from src.data.dataset import get_items, get_y_fn
from src.models.model_loader import load_config
from pathlib import Path
import matplotlib.pyplot as plt

def evaluate_model(config_path, model_path):
    """
    Evaluate a trained model on a test dataset.

    This function loads a trained model and a test dataset, and then performs evaluation by making predictions
    on the test data. It can optionally visualize the results for qualitative analysis.

    Parameters:
    - config_path (str): Path to the configuration file (config.yaml) which contains model and dataset settings.
    - model_path (str): Path to the trained model file (usually a .pkl file).

    Workflow:
    1. Load the configuration from the given path.
    2. Prepare the test dataset using DataBlock with appropriate transformations and DataLoader.
    3. Load the trained model from the specified path.
    4. Make predictions on the test dataset using the model.
    5. Optionally, visualize the results for a subset of the test dataset.
    """
    # Load Configuration
    config = load_config(config_path)

    # Prepare Test Data
    test_data = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=np.arange(config['model']['classes']))),
        get_items=get_items, 
        get_y=get_y_fn,
        item_tfms=Resize(config['data']['augmentation']['resize']),
        batch_tfms=None
    )

    test_path = Path(config['data']['path_test_dataset'])
    dls = test_data.dataloaders(test_path, bs=config['data']['batch_size'])

    # Load Model
    model = load_learner(model_path)

    # Evaluate the Model
    test_dl = dls.test_dl(get_items(test_path))
    preds, targs = model.get_preds(dl=test_dl)

    # Visualization (Optional)
    show_results(test_dl, preds, targs, config)

def show_results(test_dl, preds, targs, config, num_samples=5):
    """
    Show a comparison between original images, true masks, and predicted masks.

    Parameters:
    - test_dl: DataLoader for the test dataset.
    - preds: Predictions from the model.
    - targs: Actual targets (true masks).
    - num_samples: Number of samples to display.
    """

    # Get the path to save the figures
    figures_save_path = Path(config['paths']['figures'])

    # Get a batch of data
    x, y = test_dl.one_batch()
    for i in range(num_samples):
        # Extract the image, true mask, and predicted mask
        img = x[i]
        true_mask = y[i]
        pred_mask = preds[i].argmax(dim=0)

        # Set up the figure
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Show original image
        axs[0].imshow(img.permute(1, 2, 0))
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Show true mask
        axs[1].imshow(true_mask.squeeze(), cmap='gray')
        axs[1].set_title('True Mask')
        axs[1].axis('off')

        # Show predicted mask
        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        # Save figure
        fig.savefig(figures_save_path / f"sample_{i}_comparison.png")

        plt.show()

    print(f"Displayed {num_samples} samples.")

if __name__ == "__main__":
    evaluate_model('config.yaml', 'results/models/pspnet.pkl')

