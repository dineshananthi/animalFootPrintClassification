import uvicorn
from fastapi import FastAPI, logger
from pydantic import BaseModel

from core.vistra_afp_classification import vistra_afp_classification

APP_TITLE = "ANIMAL FOOT PRINT CLASSIFICATION"
app = FastAPI(title=APP_TITLE)


@app.get("/")
def info():
    return {"App": "Animal footprint classification"}


class Item(BaseModel):
    filePath: str


@app.post("/animal-footprint-classification")
async def do_upload_file(image_path: Item) -> dict:
    logger.logger.info("Uploading image...{}".format(image_path.filePath))
    try:
        result = vistra_afp_classification(image_path.filePath)
        return result
    except Exception as ex:
        logger.logger.error(str(ex))
        raise ex


if __name__ == "__main__":
    uvicorn.run(app, reload=True)
