## NYCU Computer Vision 2025 Spring HW4
- Student ID: 111550084
- Name: 林辰恩

### Introduction
This project implements image restoration for rainy and snowy scenes using the PromptIR framework. It supports training and testing with custom datasets and applies a combined loss function to improve restoration performance.

### How to install 👹
Clone the repository and enter its directory:
```bash
git clone https://github.com/seapa1108/NYCU-Computer-Vision-HW4
cd NYCU-Computer-Vision-HW4
```

Create and activate the Conda environment:
```bash
conda env create -f env.yml
conda activate <env_name> 
```

### Dataset Setup
Place your dataset files as follows:
```swift
data/Train/Derain/rainy/
data/Train/Derain/gt/
data/Train/Desnow/snowy/
data/Train/Desnow/gt/
```

Then update the file name lists:
```swift
data_dir/rainy/rainTrain.txt   # filenames in Derain/rainy/
data_dir/snowy/snowTrain.txt   # filenames in Desnow/snowy/
```

### Usage

Train the model
```bash
python train.py
```

Run inference on test images 
```bash
python test.py #(Place test images under test/degraded)
```
Trained checkpoints will be saved under `train_ckpt`

### Performance Snapshot
<p align="center">
  <img src="./image/hihihi.png">
</p>