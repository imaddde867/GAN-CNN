# Car Detection with a Convolutional Neural Network

This project documents the process of building a CNN to distinguish cars from other vehicles. The work is presented in the `mile_stone_2.ipynb` notebook.

The initial model performed poorly, and analysis revealed a significant class imbalance in the dataset. The final, optimized model addresses this by using class weights and regularization techniques (Dropout and L2).

The best performing model is saved in `best_model.h5`.

## Results

| Metric          | Baseline | Optimized | Improvement |
|-----------------|----------|-----------|-------------|
| Overall Accuracy| 82.7%    | 91.1%     | +8.4%       |
| Car Precision   | 50.8%    | 88.9%     | +38.1%      |
| Car Recall      | 41.0%    | 56.4%     | +15.4%      |
