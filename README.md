# Parking Segmentation

Parking Segmentation is a Python project aimed at utilizing advanced machine learning techniques for efficient and accurate segmentation of parking spaces in various environments. This project implements several state-of-the-art models to achieve its objectives.

## Features

- Implementations of PSPNet, DeepLabV3+, and U-Net models.
- Custom data augmentation techniques for robust training.
- Efficient handling of large-scale image datasets.
- Evaluation tools for model performance analysis.

## Documentation

The documentation for this project is available at the docs folder. The documentation is built using Sphinx and can be built locally using the following command:

```bash
cd docs
make html
```

For reading the documentation, execute a python server in the `docs/build/html` folder and open the `index.html` file in a browser.

```bash
cd docs/build/html
python -m http.server 8001
```

And from the browser, open http://localhost:8001