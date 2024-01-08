import yaml
import segmentation_models_pytorch as smp
from fastai.vision.learner import unet_learner
from fastai.vision.models import resnet101


def load_config(config_path):
    """
    Load the configuration from a YAML file.

    This function reads a YAML file specified by the path and loads its contents into a Python dictionary. 
    It's typically used to load configuration settings for model training, evaluation, data processing, etc.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_model(config, dls=None):
    """
    Create a model based on the configuration and (optionally) data loaders.

    This function initializes a model as specified in the configuration. It supports various types of models 
    (e.g., PSPNet, DeepLabV3+, U-Net) with specific settings like backbone and pretraining. For U-Net, 
    it additionally requires data loaders to be passed.

    Parameters:
    - config (dict): A dictionary containing model configuration, typically loaded from a YAML file.
                     It should specify the model type, backbone, pretraining, and number of classes.
    - dls (DataLoaders, optional): Data loaders required for initializing U-Net. Not needed for PSPNet or DeepLabV3+.

    Returns:
    - nn.Module: An instance of the specified neural network model.

    Raises:
    - ValueError: If the model type is unknown or if data loaders are not provided for U-Net.
    """
    model_type = config['model']['type']
    backbone = config['model']['backbone']
    pretrained = config['model']['pretrained'] == 'imagenet'
    num_classes = config['model']['classes']

    if model_type == 'pspnet':
        model = smp.PSPNet(encoder_name=backbone,
                           encoder_weights=pretrained, classes=num_classes)
    elif model_type == 'deeplabv3_plus':
        model = smp.DeepLabV3Plus(
            encoder_name=backbone, encoder_weights=pretrained, classes=num_classes)
    elif model_type == 'unet':
        if dls is None:
            raise ValueError("DataLoaders are required for fastai U-Net model")
        model = unet_learner(
            dls, resnet101, pretrained=pretrained, n_out=num_classes, metrics=[]).model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    model = create_model(config)
    print(model)
