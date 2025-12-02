# Car Classification & Detection Pipeline

`car_classification_cnn.ipynb` documents a complete workflow that starts with a binary CNN classifier for cars vs. other vehicles and culminates in a multi-class YOLO detector that is ready for Core ML deployment on Apple Silicon.

## Highlights
- Converts YOLO-style multi-class labels into a balanced binary dataset via reproducible preprocessing utilities.
- Builds a TensorFlow data pipeline (resize → normalize → shuffle → batch → prefetch) that stays GPU/MPS-friendly.
- Diagnoses class imbalance with per-class metrics, then fixes it with class weighting, dropout, L2 regularization, early stopping, checkpoints, and LR scheduling.
- Trains an Ultralytics YOLOv5-Su detector on six vehicle classes (`car`, `threewheel`, `bus`, `truck`, `motorbike`, `van`) and exports the best weights to Core ML.

## Repository Tour
| Path | Description |
| --- | --- |
| `car_classification_cnn.ipynb` | End-to-end notebook containing data prep, CNN experiments, diagnostics, YOLO training, and exports. |
| `best_model.h5` | Saved Keras model for the optimized binary classifier. |
| `yolov5su.pt` | Pretrained YOLOv5-Su checkpoint used for transfer learning. |
| `data/` | Train/validation splits with `images/` + `labels/` (YOLO txt format). |
| `runs/detect/train_optimized/` | Ultralytics training outputs, plots, metrics, and `weights/{best.pt,last.pt,best.mlpackage}`. |
| `data.yaml` | Auto-generated YOLO config that points to this dataset. |

## Dataset & Preprocessing
- Directory layout:
  ```
  data/
    train|valid/
      images/*.jpg
      labels/*.txt   # class_id x_center y_center width height
  ```
- Binary classification labels are created by reading YOLO txt files, keeping samples with class `0` (cars) and collapsing everything else into “not car”.
- Current split: 1,718 training images (20% cars) and 731 validation images (21% cars); the imbalance ratio is ~4:1 in favor of non-car vehicles.
- Images are resized to 224×224, normalized to [0, 1], mapped with `tf.data.AUTOTUNE`, shuffled (buffer 1,000), batched at 32, and prefetched for throughput.

## CNN Modeling Workflow
1. **Baseline model** – four Conv-ReLU-Pool blocks feeding two dense layers, trained for 20 epochs with Adam and no regularization. This exposes the imbalance problem: 82.7% accuracy but only 41% recall on cars (almost equivalent to predicting the majority class).
2. **Diagnostics** – confusion matrix + `classification_report` quantify the miss-rate on actual cars and guide the fixes listed below.
3. **Regularized model** – same backbone with dropout 0.4, L2 (1e-2), class weights (`car` errors count 4.0× more), and three callbacks (EarlyStopping, ModelCheckpoint → `best_model.h5`, ReduceLROnPlateau). Training runs up to 50 epochs but typically stops earlier thanks to validation monitoring.

## Binary Classification Results

| Metric | Baseline | Optimized | Δ |
| --- | --- | --- | --- |
| Overall Accuracy | 82.7% | 91.1% | +8.4 pts |
| Car Precision | 50.8% | 88.9% | +38.1 pts |
| Car Recall | 41.0% | 56.4% | +15.4 pts |

`best_model.h5` stores the checkpoint restored by EarlyStopping and is ready for inference via:
```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model("best_model.h5")
pred = model.predict(np.expand_dims(preprocessed_image, axis=0))
```

## YOLO Detection Milestone
- `data.yaml` defines six vehicle classes, and `visualize_yolo_annotation` inside the notebook overlays multiple bounding boxes per image for sanity checks.
- Training uses the Ultralytics Python API with `yolov5su.pt`, 416×416 images, batch 16, 50 epochs, patience 15, `amp=True`, `cache='disk'`, and `device='mps'` (auto-falls back to CPU/GPU).
- Best checkpoint: `runs/detect/train_optimized/weights/best.pt` (also exported as `best.mlpackage`). Final metrics: Precision 0.95, Recall 0.94, mAP50 0.978, mAP50-95 0.907.
- Visual assets such as `confusion_matrix.png`, `results.png`, and train/val batch previews live alongside the weights for easy reporting.

## Reproducing the Work
1. **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt  # or manually install tensorflow>=2.15, numpy, pandas, matplotlib, scikit-learn, ultralytics, pillow, opencv-python, torch, torchvision
   ```
2. **Run the notebook** – launch Jupyter Lab/Notebook, open `car_classification_cnn.ipynb`, and execute cells in order to rebuild the datasets, train the CNN, generate diagnostics, and kick off YOLO training/export.
3. **Train YOLO via CLI (optional)** – if you prefer commands, mirror the notebook run with:
   ```bash
   yolo detect train data=data.yaml model=yolov5su.pt imgsz=416 epochs=50 batch=16 device=mps name=train_optimized
   ```
4. **Deploy** – use `best_model.h5` for lightweight binary classification or `runs/detect/train_optimized/weights/best.pt` / `best.mlpackage` for on-device multi-class detection.

## Next Steps
- Collect more minority-class samples or augment existing car images to push recall higher.
- Fine-tune YOLO with higher-resolution crops or class-specific augmentations to lift small-motorbike performance.
- Package inference scripts (TensorFlow Serving, Core ML demo app) to showcase real-time performance.
