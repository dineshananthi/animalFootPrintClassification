import gc
import logging
import sys

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision.models import vit_b_16

from core.vistra_afp_classification import vistra_afp_classification

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

APP_TITLE = "ANIMAL FOOT PRINT CLASSIFICATION"
app = FastAPI(title=APP_TITLE)


@app.get("/")
def info():
    return {"App": "Animal footprint classification"}


class Item(BaseModel):
    filePath: str


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = '/home/dineshkumar.anandan@zucisystems.com/Downloads/vistra_animal_foot_print_class_1.pth'
    model = vit_b_16(image_size=224, num_classes=13)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logging.info(f"Loaded model from {model_path} and device: {device}")
    return model


vistra_afp_classification_model = load_model()


@app.post("/animal-footprint-classification")
async def do_upload_file(image_path: Item) -> dict:
    logging.info("Uploading image...{}".format(image_path.filePath))
    try:
        result = await vistra_afp_classification(image_path.filePath, vistra_afp_classification_model)
        return result
    except Exception as ex:
        logging.error(str(ex))
        raise ex
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    uvicorn.run(app)
