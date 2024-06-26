import gc
import io
import logging

import torch
from PIL import Image
from fastapi.logger import logger
from torchvision import transforms

from core import animal_dict


async def vistra_afp_classification(image_path: str, vistra_afp_classification_model) -> dict:
    # transformation image to tensors
    new_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        if image_path is not None:
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                # read image using pillow
                image = Image.open(image_path).convert("RGB")
                # image to tensor
                input_tensor = new_image_transform(image).unsqueeze(0).to(device)
                # disabled gradient computation for improving inference speed
                with torch.no_grad():
                    # model prediction
                    output = vistra_afp_classification_model(input_tensor)
                    del input_tensor
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, prediction_class = torch.max(output, 1)
                    confidence_score = probabilities.squeeze(0)[prediction_class].item()
                    prediction_class = prediction_class.item()
                    # starting results
                    result_dict = await result_validation(confidence_score, image_path, prediction_class)
                    logging.info(
                        "Classification completed for the image {} and output is {}".format(image_path, result_dict))
                return result_dict
            else:
                error_message = f"Invalid image extension! Please provide a valid image {image_path}"
                logger.error(error_message)
                raise ProjectException(error_message)
        else:
            logger.error("File not found {}".format(image_path))
            raise FileNotFoundError
    except Exception as exception:
        logging.error("An exception occurred {}".format(str(exception)))
        raise ProjectException(str(exception))
    finally:
        gc.collect()
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def vistra_afp_classification_ui(image_bytes, vistra_afp_classification_model) -> dict:
    # transformation image to tensors
    new_image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        if image_bytes is not None and isinstance(image_bytes, bytes):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # read image using pillow
            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream).convert("RGB")
            # image to tensor
            input_tensor = new_image_transform(image).unsqueeze(0).to(device)
            # disabled gradient computation for improving inference speed
            with torch.no_grad():
                # model prediction
                output = vistra_afp_classification_model(input_tensor)
                del input_tensor
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, prediction_class = torch.max(output, 1)
                confidence_score = probabilities.squeeze(0)[prediction_class].item()
                prediction_class = prediction_class.item()
                # starting results
                result_dict = await result_validation_ui(confidence_score, prediction_class)
                logging.info(
                    "Classification completed {}".format(result_dict))
            return result_dict
        else:
            raise FileNotFoundError
    except Exception as exception:
        logging.error("An exception occurred {}".format(str(exception)))
        raise ProjectException(str(exception))
    finally:
        gc.collect()
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def result_validation(confidence_score, image_path, prediction_class) -> dict:
    if prediction_class is not None and prediction_class >= 0:
        predicted_label = animal_dict.get(prediction_class)
        result_dict = {'inputImage': image_path if image_path is not None else predicted_label,
                       'predictedClass': prediction_class,
                       'predictedLabel': str(predicted_label).upper(),
                       'confidenceScore': "{:.2f}".format(confidence_score * 100)}
    else:
        result_dict = {'inputImage': image_path if image_path is not None else "UNKNOWN", 'predictedClass': -1,
                       'predictedLabel': "UNKNOWN",
                       'confidenceScore': 0}
    return result_dict


async def result_validation_ui(confidence_score, prediction_class) -> dict:
    if prediction_class is not None and prediction_class >= 0:
        predicted_label = animal_dict.get(prediction_class)
        result_dict = {'predictedClass': prediction_class,
                       'predictedLabel': str(predicted_label).upper(),
                       'confidenceScore': "{:.2f}".format(confidence_score * 100)}
    else:
        result_dict = {'predictedClass': -1,
                       'predictedLabel': "UNKNOWN",
                       'confidenceScore': 0}
    return result_dict


class ProjectException(Exception):
    pass
