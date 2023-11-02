import os

DEBUG = bool(os.getenv("DEBUG", False))
SECRET_KEY = os.getenv("SECRET_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./datamodels/model.pth")
