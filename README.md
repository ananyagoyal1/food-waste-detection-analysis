# Food Waste Detection & Analysis

An intelligent system to detect food vs non-food items using machine learning techniques — built to support efforts in reducing waste and promoting sustainability.

## Project Overview

This project uses a lightweight image classification model to identify food items in waste bins. The goal is to assist institutions in automating waste sorting, gathering waste statistics, and promoting responsible consumption habits. Built using **EfficientNet Lite0**, it’s optimized for fast and efficient deployment.

## Motivation

Food waste is a significant issue in college campuses, where large quantities of food are discarded on a daily basis due to over-preparation and inefficient sorting. To address this, we developed an intelligent image-based detection system that can identify food items in waste bins—making it easier to measure, monitor, and ultimately reduce food waste across campus dining facilities. By using machine learning for smarter waste management, we aim to foster sustainability and raise awareness within student communities.

## Features

- Classifies images as **food** or **non-food**
- Trained on a custom dataset using transfer learning
- Fast and portable inference using **TensorFlow Lite**
- Includes Jupyter notebook for experiment tracking

## Repository Structure

1. ├── 2022-03-18_food_not_food_model_efficientnet_lite0_v1.tflite  # Trained ML model
2. ├── Copy_of_Untitled9gfg.ipynb                                   # Notebook for testing/analysis
3. ├── LICENSE                                                      # MIT License
4. ├── README.md                                                    # Project overview
5. ├── audino code/                                                 # Annotation tool setup
6. ├── food_dataset_neww.zip                                        # Image dataset archive
7. ├── food_not_food.py                                             # Inference script
8. ├── hardware setup.jpeg                                          # Hardware configuration image


## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/ananyagoyal1/food-waste-detection-analysis.git
   cd food-waste-detection-analysis

2. **Install dependencies**
   pip install tensorflow numpy matplotlib
   
3. **Run the model**
   python food_not_food.py

## Dataset

The dataset contains labeled food and non-food images. It’s packaged in a ZIP archive named `food_dataset_neww.zip`, which includes the training images used for model development. You can use the **Audino tool** (available in the `audino code/` folder) to annotate new images or extend the dataset for further training.

To create a custom dataset:

1. Organize your images into two folders:
   dataset/ ├── food/ └── non_food/

2. Use Audino or any annotation tool to label image samples if needed.

3. Preprocess images to match the input shape of the EfficientNet Lite0 model (e.g., 224x224 pixels).

## Model Details

- **Architecture:** EfficientNet Lite0
- **Framework:** TensorFlow / TensorFlow Lite
- **Input:** Preprocessed image (size: 224x224)
- **Output:** Binary classification - `food` or `non-food`
- **Format:** Saved model in `.tflite` for edge-device inference

## License

This project is licensed under the [MIT License](./LICENSE).
