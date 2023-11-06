import os

from core.schemas import TrainingStatus

DEBUG = bool(os.getenv("DEBUG", False))
SECRET_KEY = os.getenv("SECRET_KEY", "")

# Dataset
DATASET_FOLDER = os.getenv("DATASET_FOLDER", "./datasets/")
DATASET_TRAIN_FOLDER_NAME = os.getenv("DATASET_TRAIN_FOLDER_NAME", "train")
DATASET_VALID_FOLDER_NAME = os.getenv("DATASET_VALID_FOLDER_NAME", "val")
DATASET_ANNOTATION_FILE = os.getenv("DATASET_ANNOTATION_FILE", "classes.json")

# Dataset Model
DATASET_MODEL_FOLDER = os.getenv("DATASET_MODEL_FOLDER", "./datamodels/")
DATASET_MODEL_NAME = os.getenv("DATASET_MODEL_NAME", "model.pth")

# Backup
BACKUPS_FOLDER = os.getenv("BACKUPS_FOLDER", "./backups/")

# Cloud values
CLOUD_TYPE = os.getenv("CLOUD_TYPE", None)
CLOUD_ID = os.getenv("CLOUD_ID", None)
CLOUD_TOKEN = os.getenv("CLOUD_TOKEN", None)
CLOUD_SECRET = os.getenv("CLOUD_SECRET", None)
CLOUD_TRAIN_DATASET_PATH = os.getenv("CLOUD_TRAIN_DATASET_PATH", None)

# DB
DB_STATUS_FILE_NAME = os.getenv("DB_STATUS_FILE_NAME", "status_db")

# Auto settings
STATUS = TrainingStatus.READY

DATASET_CLASSES_PATH = os.path.join(DATASET_FOLDER, DATASET_ANNOTATION_FILE)
LOCAL_TRAIN_DATASET_PATH = os.path.join(
    DATASET_FOLDER, DATASET_TRAIN_FOLDER_NAME
)
LOCAL_VALID_DATASET_PATH = os.path.join(
    DATASET_FOLDER, DATASET_VALID_FOLDER_NAME
)

DATASET_MODEL_PATH = os.path.join(DATASET_MODEL_FOLDER, DATASET_MODEL_NAME)
DATASET_MODEL_STATUS_DB_PATH = os.path.join(
    DATASET_MODEL_FOLDER, DB_STATUS_FILE_NAME
)
