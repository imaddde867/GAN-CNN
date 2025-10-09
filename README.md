# GAN-CNN Vehicle Detection

Vehicle detection using GANs and CNNs. Detects 6 vehicle types: cars, trucks, buses, motorbikes, three-wheelers, and vans.

## Dataset

Download the dataset and extract it to a `data` folder:
https://storage.googleapis.com/kaggle-data-sets/2701470/4649306/bundle/archive.zip

Structure should be:
```
data/
├── train/images/  (2100 images)
├── train/labels/  (YOLO format annotations)
├── valid/images/  (900 images)
└── valid/labels/
```

## Classes
- 0: car
- 1: threewheel  
- 2: bus
- 3: truck
- 4: motorbike
- 5: van

## Next is

1. Extract dataset to `data/` folder
2. Run `main.ipynb` to load and process the data
3. Labels are in YOLO format: `class_id x_center y_center width height` (normalized coordinates)

So far the notebook loads the files and shows dataset statistics.