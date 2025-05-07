from fastapi import FastAPI
from app.routers.app_routers import router

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Backend API of the Agent",
        "docs": "/docs",
        "api_base": "/api"
    }

app.include_router(router, prefix="/api", tags=["api"])