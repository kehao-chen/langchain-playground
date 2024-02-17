from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
llm = ChatOpenAI(openai_api_key=settings.openai_api_key)
