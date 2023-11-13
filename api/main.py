from fastapi import FastAPI, UploadFile

from nnetwork.main import FakeDetector

app = FastAPI()


@app.post("/detect/")
async def create_upload_file(file: UploadFile):
    """:return
    """
    nn = FakeDetector(file)
    return {"filename": file.filename, "prediction": nn.prediction}
