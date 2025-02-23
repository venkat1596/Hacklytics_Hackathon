from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from test_script_for_cut import RunModel
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
def ProcessFile(image):
    pass        # Going to call the file image proccess

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/send-and-rec")
async def send_and_rec(image: UploadFile = File(...)):
    # makes the path to store the upload
    in_path = f"{UPLOAD_FOLDER}/{image.filename}"
    # Save the uploaded image
    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    modelPath = os.getcwd() + "/pths/latest_net_G.pth" ###MUST CHANGE THIS MANUALLY
    #Constructs model path without using hardcoding
    outputUrl = RunModel(in_path, modelPath, 'result.jpg')
    
    return FileResponse(outputUrl)

    # print out file response to see what the format is like