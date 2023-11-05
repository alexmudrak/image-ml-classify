import os

DEBUG = bool(os.getenv("DEBUG", False))
SECRET_KEY = os.getenv("SECRET_KEY", "")
DATAMODEL_PATH = os.getenv("DATAMODEL_PATH", "./datamodels/model.pth")
DATASETS_FOLDER = os.getenv("DATASETS_FOLDER", "./datasets/")
BACKUPS_FOLDER = os.getenv("BACKUPS_FOLDER", "./backups/")
