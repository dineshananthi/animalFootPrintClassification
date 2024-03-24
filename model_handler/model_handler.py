import torch
from fastapi.logger import logger
from torchvision.models.vision_transformer import vit_b_16

from model_handler import default_image_size, number_of_classes

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LoadModel:

    @staticmethod
    def vistra_model():
        model_path = "/home/dineshkumar.anandan@zucisystems.com/PycharmProjects/animalFootPrintClassification/model_handler/model/vistra_animal_foot_print_classification.pth"
        model = vit_b_16(image_size=default_image_size, num_classes=number_of_classes)
        model = model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logger.info("Animal foot print classification model loaded using {}".format(device))
        return model
