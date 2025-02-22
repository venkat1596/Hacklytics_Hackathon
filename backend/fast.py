from fastapi import FastAPI
from fastapi import File, UploadFile, FileResponse



app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/send-and-rec")
async def send_and_rec(image: UploadFile = File(...)):
    
    return {FileResponse(ProcessImage(image))}