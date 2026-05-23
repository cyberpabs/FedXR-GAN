import os

TF_DEVICE = "/device:GPU:0"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets')

DATASET_OPTIONS = {
    "chest-xray-pneumonia": {
        "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia",
        "path": os.path.join(DATASETS_DIR, "chest-xray-pneumonia")
    },
    "ct-to-mri-brain-scans": {
        "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/darren2020/ct-to-mri-cgan",
        "path": os.path.join(DATASETS_DIR, "ct-to-mri-cgan")
    }
}

DATASET_DATA = {
    "chest-xray-pneumonia": {
        "path": f'{DATASET_OPTIONS["chest-xray-pneumonia"]["path"]}/chest_xray/train/NORMAL',
        "extension": 'jpeg'
    },
    "ct-to-mri-brain-scans": {
        "path": f'{DATASET_OPTIONS["ct-to-mri-brain-scans"]["path"]}/Dataset/images/trainA', # CT scans
        "extension": 'png'
    }
}

PREPROCESSED_IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32 # For laptop GPU
BUFFER_SIZE = 60000

NUM_EXAMPLES_TO_GENERATE = 16