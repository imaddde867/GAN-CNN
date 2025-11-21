# Car Detection with a Convolutional Neural Network

This project documents the process of building a CNN to distinguish cars from other vehicles. The work is presented in the `car_classification_cnn.ipynb` notebook.

The initial model performed poorly, and analysis revealed a significant class imbalance in the dataset. The final, optimized model addresses this by using class weights and regularization techniques (Dropout and L2).

The best performing model is saved in `best_model.h5`.

## Dataset

The data is organized into `train` and `valid` sets, each containing `images` and `labels` subdirectories. The labels are in YOLO format.

For this project, the multi-class problem was simplified to a binary classification task: identifying whether a vehicle is a 'car' (class 0) or 'not a car'.

The dataset is highly imbalanced, with 'not a car' samples significantly outnumbering 'car' samples. This imbalance was a key challenge addressed in the modeling process.

## Dataset Preview

Here is a small preview of the images in the dataset, showing examples of the "car" and "not-car" classes.

### Cars
<img src="data/train/images/00077.jpg" width="200"/> <img src="data/train/images/00116.jpg" width="200"/> <img src="data/train/images/00241.jpg" width="200"/>

### Not-Cars (Vans, Trucks, etc.)
<img src="data/train/images/00043_GMC%20Savana%20Van%202012.jpg" width="200"/> <img src="data/train/images/00175_Ford%20E-Series%20Wagon%20Van%202012.jpg" width="200"/> <img src="data/train/images/indian-truck-lorry-on-highway-260nw-1987912226.jpg" width="200"/>

## Results

| Metric          | Baseline | Optimized | Improvement |
|-----------------|----------|-----------|-------------|
| Overall Accuracy| 82.7%    | 91.1%     | +8.4%       |
| Car Precision   | 50.8%    | 88.9%     | +38.1%      |
| Car Recall      | 41.0%    | 56.4%     | +15.4%      |