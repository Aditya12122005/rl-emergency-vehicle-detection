# How to Download the Dataset

The dataset for this project originates from Roboflow Universe. While the exact download link is not available in the project files, you can find and download a suitable dataset by following these steps.

## 1. Find a Dataset on Roboflow Universe

1.  Go to [Roboflow Universe](https://universe.roboflow.com/).
2.  Search for "emergency vehicle detection".
3.  Look for a dataset that has a variety of classes. The original dataset used for this project had 24 classes, including: `Army`, `Vehicle`, `ambulance`, `ambulance_108`, `ambulance_SOL`, `ambulance_lamp`, `ambulance_text`, `auto`, `bike`, `bus`, `car`, `fire_truck`, `fireladder`, `firelamp`, `firesymbol`, `firewriting`, `horse`, `police`, `police_lamp`, `police_lamp_ON`, `road_sign`, `tempo traveller`, `truck`, `writing`. You may need to find a dataset with similar classes and adapt the `data/data.yaml` file accordingly.

## 2. Download the Dataset

Once you have found a suitable dataset on Roboflow, you can download it in the "YOLOv8" format. This will give you a zip file.

## 3. Place the Dataset in the Correct Directory

1.  Unzip the downloaded file.
2.  You should see `train`, `valid`, and `test` folders, along with a `data.yaml` file.
3.  Move the contents of the unzipped `train`, `valid`, and `test` folders into the `data/train`, `data/valid`, and `data/test` directories of this project, respectively.
4.  You may need to update the `data/data.yaml` file in this project to reflect the class names and number of classes of the dataset you downloaded.
