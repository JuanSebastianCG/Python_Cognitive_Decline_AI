from fastapi import HTTPException, Depends, Security
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()

def get_db(settings: Settings = Depends(get_settings)):
    # Aquí podrías crear y retornar una conexión de base de datos
    pass
