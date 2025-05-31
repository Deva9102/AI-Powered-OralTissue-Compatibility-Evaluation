# AI-Powered-OralTissue-Compatibility-Evaluation

This project presents an AI-powered solution to assess oral tissue compatibility for dental implants using ultrasonography images. By combining deep learning and traditional machine learning techniques, the system automates the classification of tissue for implants.


## Overview

- Ultrasound images are preprocessed with normalization, grayscale conversion, and smoothing.
- Augmentation (flip, rotate, enhance) increases dataset diversity.
- Feature extraction uses both raw pixel intensities and HOG descriptors.
- Models used:
  - Traditional ML: SVM, Random Forest, Gradient Boosting, Decision Tree
  - Deep Learning: Custom ResNet, DBN-style MLP, Hybrid ResNet+DBN
- Grad-CAM is used to highlight regions contributing to predictions.
- The system can classify new images with clear "Suitable"/"Not Suitable" predictions.
