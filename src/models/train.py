from fastai.vision.all import *
from src.utils.metrics import save_metrics_to_csv
from src.data.dataset import get_items, get_y_fn
from src.models.model_loader import load_config, create_model
from fastai.vision.augment import aug_transforms
from fastai.data.transforms import Normalize
from src.utils.transforms import ShadowTransform
from torchvision.transforms import Resize


def train_model(config_path, model_type):
    """
    Conduct the training process for a deep learning model.

    This function orchestrates the training pipeline, including data preparation, model initialization,
    training, and saving the trained model and metrics. It relies on a configuration file for all settings.

    Parameters:
    - config_path (str): Path to the configuration file (config.yaml), which contains settings for data,
                         model, training, and paths for saving outputs.
    - model_type (str): Type of the model to train ('pspnet', 'deeplabv3_plus', 'unet').

    Steps:
    1. Load configuration from the given path.
    2. Set up data augmentation transformations based on configuration.
    3. Prepare the data using FastAI's DataBlock API.
    4. Initialize the model specified in the configuration.
    5. Create a FastAI Learner for training.
    6. Conduct the training process.
    7. Save the trained model and metrics in specified paths.
    """
    # Load Configuration
    config = load_config(config_path)
    print("Configuration loaded successfully.")

    # Validate model type
    if model_type not in config['model']['type']:
        raise ValueError(f"Model type '{model_type}' is not supported. Choose from {config['model']['type']}")
    print(f"Selected model type: {model_type}")

    # Update model type in the configuration
    config['model']['type'] = model_type

    # Data Augmentation Setup
    batch_tfms = setup_augmentations(config['data']['augmentation'])
    print("Data augmentation setup completed.")

    # Data Preparation
    data = DataBlock(
        blocks=(ImageBlock, MaskBlock(
            codes=np.arange(config['model']['classes']))),
        get_items=get_items,
        get_y=get_y_fn,
        splitter=RandomSplitter(
            valid_pct=config['data']['validation_split'], seed=42),
        item_tfms=Resize(config['data']['augmentation']['resize']),
        batch_tfms=batch_tfms
    )

    dls = data.dataloaders(
        config['data']['path_to_dataset'], bs=config['data']['batch_size'])
    print("Data preparation completed.")

    # Model Initialization
    model = create_model(config, dls)
    print(f"Model '{config['model']['type']}' initialized.")

    # Create Learner
    learner = Learner(dls, model, loss_func=FocalLoss(), metrics=[
                      foreground_acc, DiceMulti(), JaccardCoeffMulti()], 
                      cbs=[ShowGraphCallback()])
    print("Learner created, starting training process.")
    
    # Training
    print("Starting training...")
    learner.fit_one_cycle(config['training']['epochs'])
    print("Training completed.")

    # Save model
    model_save_path = Path(config['paths']['models']) / \
        f"{config['model']['type']}_model.pkl"
    learner.export(fname=model_save_path)
    print(f"Model saved at {model_save_path}")

    # Save metrics
    metrics_save_path = Path(
        config['paths']['metrics']) / f"{config['model']['type']}_metrics.csv"
    save_metrics_to_csv(learner, file_path=metrics_save_path)
    print(f"Metrics saved at {metrics_save_path}")
    print("Training process completed and outputs saved.")


def setup_augmentations(aug_config):
    """
    Set up data augmentation transformations based on a configuration dictionary.

    This function builds a list of transformations including resizing, custom shadow transformation, 
    and other augmentations like flips, rotations, zooming as specified in the configuration.

    Parameters:
    - aug_config (dict): A dictionary containing augmentation settings. This includes details for
                         resizing, shadow transformation, and other augmentations like flips, rotations, etc.

    Returns:
    - List[Transform]: A list of FastAI Transform objects to be applied to the dataset.
    """
    # Initialize a list to hold the transformations
    transforms_list = []

    # Resize transformation
    if 'resize' in aug_config:
        transforms_list.append(Resize(aug_config['resize']))

    # Shadow transformation
    if 'shadow_transform' in aug_config:
        shadow_params = aug_config['shadow_transform']
        transforms_list.append(ShadowTransform(num_shadows=shadow_params['num_shadows'],
                                               min_opacity=shadow_params['min_opacity'],
                                               max_opacity=shadow_params['max_opacity']))

    # Fastai's aug_transforms
    if 'aug_transforms' in aug_config:
        aug_params = aug_config['aug_transforms']
        transforms_list.extend(aug_transforms(mult=aug_params['mult'],
                                              do_flip=aug_params['do_flip'],
                                              flip_vert=aug_params['flip_vert'],
                                              max_rotate=aug_params['max_rotate'],
                                              min_zoom=aug_params['min_zoom'],
                                              max_zoom=aug_params['max_zoom'],
                                              max_lighting=aug_params['max_lighting'],
                                              max_warp=aug_params['max_warp'],
                                              p_affine=aug_params['p_affine'],
                                              p_lighting=aug_params['p_lighting'],
                                              xtra_tfms=aug_params['xtra_tfms'] if 'xtra_tfms' in aug_params else None,
                                              size=aug_params['size'] if 'size' in aug_params else None,
                                              mode=aug_params['mode'],
                                              pad_mode=aug_params['pad_mode'],
                                              align_corners=aug_params['align_corners'],
                                              batch=aug_params['batch'],
                                              min_scale=aug_params['min_scale']))

    # Normalization
    if 'normalize' in aug_config:
        transforms_list.append(Normalize.from_stats(*imagenet_stats))

    return transforms_list


if __name__ == "__main__":
    train_model('config.yaml', 'pspnet')
