import os


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/model")
    DEVICE = os.getenv("DEVICE", "cuda:0")
    NUM_GPUS = int(os.getenv("NUM_GPUS", 4))  # Changed default to 4
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    USE_SAFETENSORS = os.getenv("USE_SAFETENSORS", "True").lower() in ["true", "1", "t"]

    @staticmethod
    def get_device_map():
        if Config.NUM_GPUS > 1:
            return "auto"
        return Config.DEVICE

    @staticmethod
    def get_model_weights_path():
        return os.path.join(Config.MODEL_PATH, "weights")