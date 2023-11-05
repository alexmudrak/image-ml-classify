from enum import Enum


class TrainingStatus(Enum):
    READY = "Ready"
    DATASET_SYNCHRONIZATION = "Sync dataset"
    MODEL_TRAINING = "Model training"
    ERROR = "Error"
