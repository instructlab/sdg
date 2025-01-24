# Standard
from importlib import resources
import pathlib
import typing

# Third Party
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings
from openai import OpenAI
from starlette.testclient import TestClient


def llama_cpp_openai_client(model, model_repo_id):
    server_settings = ServerSettings()
    model_settings = [
        ModelSettings(
            model=model,
            hf_model_repo_id=model_repo_id,
            verbose=True,
        )
    ]
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )

    @app.get("/")
    def read_root():
        return {"message": "Hello from InstructLab! Visit us at https://instructlab.ai"}

    test_client = TestClient(app)
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        http_client=test_client,
    )
