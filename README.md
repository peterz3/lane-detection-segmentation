# lane-detection-segmentation

### Installing

Install with

```shell
git clone https://github.com/divamgupta/image-segmentation-keras
cd image-segmentation-keras
python setup.py install
```

### Training

Train the model( not neccessary if you cloned the repo as pretrained model has been saved)

```shell
python3 -m train --checkpoints_path="path_to_checkpoints"  --train_images="train_image_location"  --train_annotations="label_locations"   --n_classes=2  --input_height=720  --input_width=1280  --model_name="segnet"
```

### Predicting

Predict lane location on a image

```shell
python3 predict.py --checkpoints_path="path_to_checkpoints" --input_path="input_image" --output_path="path_to_predictions/output_path.png" 
```
