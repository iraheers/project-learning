from fastapi import FastAPI, Depends, HTTPException, Header
import ollama
import os
from dotenv import load_dotenv
from PIL import Image


load_dotenv() #loads value from environment variable file
API_KEY_CREDITS = {os.getenv("API_KEY"): 5}

app = FastAPI()

def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEY_CREDITS.get(x_api_key, 0)

    if credits <=0:
        pass
        #raise HTTPException(status_code = 401, detail="unauthorised")

    return x_api_key

@app.post("/generate")
def generate(prompt: str, x_api_key: str = Depends(verify_api_key)):
    API_KEY_CREDITS[x_api_key]-=1
    
    response = ollama.chat(model="llava", messages=[{"role":"user", "content":prompt}])

    return {"response": response["message"]["content"]}


@app.post("/generate")
def generate(prompt: str, x_api_key: str = Depends(verify_api_key)):
    API_KEY_CREDITS[x_api_key]-=1
    
    response = ollama.chat(model="llava", messages=[{"role":"user", "content":prompt}])

    return {"response": response["message"]["content"]}


@app.post("/get-caption")
async def get_caption(file_path: str = None,):

    # instruction = """Identify all the food items visible in this image. Be very specific and detailed for each item.
    # -For each food item, describe:What it is (e.g., "pepperoni pizza", "caesar salad", "spaghetti carbonara")
    # - If it's a combination (e.g., burger with cheese and lettuce), describe all components
    # - If it's a drink or sauce, mention that too
    # - Ignore non-food objects.
    # List each food item separately, clearly and accurately."""

    instruction = """What is in the image?"""

    result = ollama.generate(
        model='llava',
        prompt=instruction,
        images=[file_path],
        stream=False
    )['response']
    
    return {"result": result}



@app.post("/test")
async def test(file_path: str = None,):
    return {"result": "hello"}
    