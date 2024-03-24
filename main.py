import uvicorn
from fastapi import FastAPI, logger
from pydantic import BaseModel

APP_TITLE = "ANIMAL FOOT PRINT CLASSIFICATION"
app = FastAPI(title=APP_TITLE)


@app.get("/")
def info():
    return {"App": "Animal footprint classification"}


class Item(BaseModel):
    filePath: str


@app.post("/animal-footprint-classification")
async def do_upload_file(image_path: Item):
    return None


if __name__ == "__main__":
    uvicorn.run(app, reload=True)
