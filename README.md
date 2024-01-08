# Diabetic Retinopathy Detection with AI

## Setup

### Gradio app environment

TODO

### Training environment

Create conda environment from YAML:
```bash
mamba env create -n retinopathy_train -f environment.yml
```

Download the data from [Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) or use kaggle API:

```bash
pip install kaggle
kaggle competitions download -c diabetic-retinopathy-detection
mkdir retinopathy_data/
unzip diabetic-retinopathy-detection.zip -d retinopathy_data/
```

Launch training:
```bash
conda activate retinopathy_train
python train.py
```
The trained model will be put into `lightning_logs/`.