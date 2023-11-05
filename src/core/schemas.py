from enum import Enum


class TrainingStatus(Enum):
    READY = "Model Ready for Use"
    DATASET_SYNCHRONIZATION = "Dataset Synchronization in Progress"
    MODEL_TRAINING = "Model Training in Progress"
    ERROR = "Error"
