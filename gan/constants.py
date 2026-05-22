# Adjust the following constants to your needs

TF_DEVICE = "/device:GPU:0"

DATASET_OPTIONS = {
    "chest-xray-pneumonia": {
        "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia",
        "path": "./datasets/chest-xray-pneumonia"
    },
    "ct-to-mri-brain-scans": {
        "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/darren2020/ct-to-mri-cgan",
        "path": "./datasets/ct-to-mri-cgan"
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

PREPROCESSED_IMAGE_SIZE = (64, 64) #(128, 128)
BATCH_SIZE = 256
BUFFER_SIZE = 60000

NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

EPOCHS = 7000
CHECKPOINT_DIR = './training_checkpoints'