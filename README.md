# Product Image Quality Predictor

A lightweight image-quality classifier that automatically detects low-quality product images (blur, noise, resolution loss, lighting issues) so only high-quality visuals are used in production. Built with transfer learning on a ResNet50 backbone and trained on a synthetic dataset derived from the Kaggle "fashion-product-images-small" dataset.

Live notebook: Product_Image_quality_predictor.ipynb

## Highlights / What this project does

- Generates a balanced dataset of "good" and "bad" product images by sampling good images and synthetically degrading copies to create bad ones.
- Uses a pre-trained ResNet50 (ImageNet weights) as a frozen feature extractor and a small classifier head to classify images as Good / Bad.
- Fast to iterate: trains in minutes per epoch on GPU (notebook used a T4 in Colab).
- Provides an inference helper to visualize and predict quality for individual images.

## Dataset and Data Preparation (SOW flow)

1. Source: Kaggle dataset `paramaggarwal/fashion-product-images-small` (MIT license).
2. Extraction: downloaded and extracted the dataset in the Colab notebook.
3. Sampling: randomly sampled 10,000 "good" images from the dataset to create the base set.
4. Synthetic negative generation: created 10,000 "bad" images by applying random combinations of degradations:
   - Gaussian blur (kernel sizes 3, 5, 7)
   - Additive Gaussian noise
   - Reduced JPEG quality (quality in range 10-40)
   - Downscale-upscale (resolution reduction)
   - Random brightness changes
5. Final dataset structure: `dataset/good_images` and `dataset/bad` then organized into `dataset/` for ImageDataGenerator with a 80/20 train/validation split.

This SOW (scope-of-work) yields a realistic supervised dataset for image-quality classification with equal representation for both classes.

## Model architecture and training details

- Backbone: ResNet50 (pre-trained on ImageNet) with `include_top=False` and input shape (256,256,3).
- Trainable layers: backbone frozen (all layers set to non-trainable) for fast fine-tuning of the head.
- Classifier head:
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(1, activation='sigmoid')
- Loss: binary_crossentropy
- Optimizer: Adam(learning_rate=1e-4)
- Metrics: accuracy (binary)
- Input preprocessing: rescale=1./255, images resized to 256x256
- Batch size: 32
- Epochs in experiment: 5 (used as a quick proof-of-concept)

## Results / Metrics (from experiment)

- Dataset used for training/validation during the notebook run: ~16,000 training images and ~4,000 validation images (80/20 split on 20k total images).
- Observed training accuracy after 5 epochs: ~82% (training logs: accuracy progressed from ~62% -> ~78% -> ~81% -> ~80% -> ~82%).

Note: these values are from a short training run used as a proof-of-concept in the notebook. For robust production performance, consider:
- Longer training (more epochs) with staged unfreezing of the backbone
- Stronger augmentation and class of degradations matching your production failure modes
- Hyperparameter search (learning rates, dropout, head size)
- Evaluation with precision, recall, F1 and confusion matrix on a held-out test set

## How to run (Colab)

1. Open the notebook: `Product_Image_quality_predictor.ipynb` (Colab link is embedded in the notebook header).
2. Install dependencies (already provided in notebook):
   - tensorflow
   - opencv-python
   - kaggle (for dataset download)
3. Upload your `kaggle.json` credentials to access the Kaggle dataset or provide your own images.
4. Run the cells top-to-bottom. The notebook downloads the dataset, builds the synthetic negatives, prepares generators, builds the model, trains and demonstrates predictions.

## Inference

The notebook provides a helper `predict_output(image_path)` which:
- Loads and resizes the image to 256x256
- Normalizes the image (scale 0..1)
- Feeds the image to the trained model and prints the prediction probability and label (Good if prob > 0.5)

Example usage (already in the notebook):

predict_output("/content/dataset/good_images/15189.jpg")

## Files in this repository

- Product_Image_quality_predictor.ipynb — full notebook with dataset download, synthetic bad image generation, training and inference examples.
- README.md — this file.

## Requirements

- Python 3.8+ (notebook used Python 3 in Colab)
- tensorflow
- opencv-python
- keras (bundled with tensorflow)
- kaggle (optional, for downloading the Kaggle dataset)

Install quick requirements (use pip or Colab cells in the notebook):

pip install tensorflow opencv-python kaggle

## Next steps / Suggestions for improvement

- Expand dataset diversity: include real-world low-quality images from your platform to reduce sim/real gap.
- Unfreeze some ResNet50 layers and fine-tune with a lower learning rate to improve generalization.
- Replace the binary classifier with a regression or multi-label model to predict continuous quality scores (e.g., 0-100) or multiple quality issues (blur/noise/exposure).
- Add comprehensive evaluation: compute ROC AUC, precision/recall, F1, and show a confusion matrix on a held-out test set.
- Deploy as a lightweight inference service (FastAPI / TensorFlow Serving / ONNX) to perform real-time checks during image upload.

## License

This repository is provided as-is. The dataset used in the notebook (`fashion-product-images-small`) is under the MIT license — check the dataset page for terms.

## Contact

Author: Khushi Keswani

If you want help extending this project (regression scoring, more degradations, deployment), open an issue or submit a PR.
