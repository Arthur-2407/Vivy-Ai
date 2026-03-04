import os
import json
from models.download import download_file

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIG_PATH = os.path.join(BASE_DIR, "config", "models.json")


def ensure_model(model_name):

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model_info = config[model_name]

    model_path = os.path.join(MODEL_DIR, model_info["file"])

    if not os.path.exists(model_path):

        print(f"Downloading {model_name} model...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        download_file(model_info["url"], model_path)

        print("Download complete.")

    else:
        print(f"{model_name} model already exists.")

    return model_path