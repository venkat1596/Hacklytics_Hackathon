from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
#model = Model()
def ProcessFile(image):
    pass        # Going to call the file image proccess

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/send-and-rec")
async def send_and_rec(image: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{image.filename}"
    print("entered send_and_rec")
    # Save the uploaded image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return FileResponse(file_path)

    # print out file response to see what the format is like