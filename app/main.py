from fastapi import FastAPI
from app.routers import endpoints
from app.dependencies import get_settings

app = FastAPI()

app.include_router(endpoints.router)

@app.get("/")
def read_root():
    return {"Hello": "Welcome to the Cognitive API"}
