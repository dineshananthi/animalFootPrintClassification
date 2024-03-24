import gc
import logging

import torch
from PIL import Image
from fastapi.logger import logger
from torchvision import transforms

from core import animal_dict
from model_handler import default_image_size
from model_handler.model_handler import LoadModel

# singleton load
vistra_afp_classification_model = LoadModel.vistra_model()


def vistra_afp_classification(image_path: str) -> dict:
    new_image_transform = transforms.Compose([
        transforms.Resize((default_image_size, default_image_size)),
        transforms.ToTensor(),
    ])
    try:
        if image_path is not None:
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                image = Image.open(image_path).convert("RGB")
                input_tensor = new_image_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():  # disabled gradient computation for improving inference speed
                    output = vistra_afp_classification_model(input_tensor)
                    del input_tensor
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, prediction_class = torch.max(output, 1)
                    confidence_score = probabilities.squeeze(0)[prediction_class].item()
                    prediction_class = prediction_class.item()

                    if 0 <= prediction_class >= 12 and prediction_class is not None:
                        predicted_label = animal_dict[prediction_class]
                        result_dict = {'inputImage': image_path, 'predictedClass': prediction_class,
                                       'predictedLabel': str(predicted_label).upper(),
                                       'confidenceScore': "{:.2f}".format(confidence_score * 100)}
                    else:
                        result_dict = {'inputImage': image_path, 'predictedClass': -1,
                                       'predictedLabel': "UNKNOWN",
                                       'confidenceScore': 0}
                return result_dict
            else:
                error_message = f"Invalid image extension! Please provide a valid image {image_path}"
                logger.error(error_message)
                raise error_message
        else:
            logger.error("File not found {}".format(image_path))
            raise FileNotFoundError
    except Exception as exception:
        logging.error("An exception occurred {}".format(str(exception)))
        print(exception)
    finally:
        gc.collect()
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
