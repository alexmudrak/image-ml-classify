import os

from core.schemas import TrainingStatus

DEBUG = bool(os.getenv("DEBUG", False))
SECRET_KEY = os.getenv("SECRET_KEY", "")
DATAMODEL_PATH = os.getenv("DATAMODEL_PATH", "./datamodels/model.pth")
DATASETS_FOLDER = os.getenv("DATASETS_FOLDER", "./datasets/")
BACKUPS_FOLDER = os.getenv("BACKUPS_FOLDER", "./backups/")


# Cloud values
CLOUD_TYPE = os.getenv("CLOUD_TYPE", None)
CLOUD_ID = os.getenv("CLOUD_ID", None)
CLOUD_TOKEN = os.getenv("CLOUD_TOKEN", None)
CLOUD_SECRET = os.getenv("CLOUD_SECRET", None)
CLOUD_TRAIN_DATASET_PATH = os.getenv("CLOUD_TRAIN_DATASET_PATH", None)

# DB
DB_STATUS_FILE = os.getenv("DB_STATUS_FILE", "status_db")

# Auto settings
STATUS = TrainingStatus.READY
LOCAL_TRAIN_DATASET_PATH = os.path.join(DATASETS_FOLDER, "train")
