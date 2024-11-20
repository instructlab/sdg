# Standard
from importlib import resources
import pathlib
import typing

# Third Party
from datasets import Dataset
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings
from openai import OpenAI
from starlette.testclient import TestClient
import pytest

# First Party
from src.instructlab.sdg.pipeline import Pipeline, PipelineContext


TESTS_PATH = pathlib.Path(__file__).parent.parent.absolute()


@pytest.fixture
def testdata_path() -> typing.Generator[pathlib.Path, None, None]:
    """Path to local test data directory"""
    yield TESTS_PATH / "testdata"


@pytest.fixture
def num_gpu_layers():
    return 0


@pytest.fixture
def openai_client(model, model_repo_id, num_gpu_layers):
    server_settings = ServerSettings()
    model_settings = [
        ModelSettings(
            model=model,
            hf_model_repo_id=model_repo_id,
            n_gpu_layers=num_gpu_layers,  # just run on the CPU
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


@pytest.fixture
def teacher_model(openai_client):
    models = openai_client.models.list()
    return models.data[0].id


@pytest.fixture
def max_num_tokens():
    return 256


@pytest.fixture
def pipeline_context(
    openai_client,
    model_family,
    teacher_model,
    num_instructions_to_generate,
    max_num_tokens,
):
    return PipelineContext(
        openai_client,
        model_family,
        teacher_model,
        num_instructions_to_generate,
        max_num_tokens=max_num_tokens,
    )


@pytest.fixture
def knowledge_dataset():
    return Dataset.from_list(
        [
            {
                "icl_query_1": "what is the location of the tubal tonsils?",
                "icl_response_1": "The location of the tubal tonsils is the roof of the pharynx.",
                "icl_query_2": "How long does the adenoid grow?",
                "task_description": "Teaching about human anatomy, specifically tonsils",
                "icl_response_2": "The adenoid grows until the age of 5, starts to shrink at the age of 7 and becomes small in adulthood.",
                "icl_query_3": "What is the immune systems first line of defense against ingested or inhaled foreign pathogens?",
                "icl_response_3": "The tonsils are the immune systems first line of defense.",
                "document": "The **tonsils** are a set of lymphoid organs facing into the aerodigestive tract, which is known as Waldeyer's tonsillar ring and consists of the adenoid tonsil or pharyngeal tonsil, two tubal tonsils, two palatine tonsils, and the lingual tonsils. These organs play an important role in the immune system. When used unqualified, the term most commonly refers specifically to the palatine tonsils, which are two lymphoid organs situated at either side of the back of the human throat. The palatine tonsils and the adenoid tonsil are organs consisting of lymphoepithelial tissue located near the oropharynx and nasopharynx parts of the throat",
                "icl_document": "The **tonsils** are a set of lymphoid organs facing into the aerodigestive tract, which is known as Waldeyer's tonsillar ring and consists of the adenoid tonsil or pharyngeal tonsil, two tubal tonsils, two palatine tonsils, and the lingual tonsils.",
                "domain": "textbook",
                "document_outline": "Medical description of tonsils",
            }
        ]
    )


@pytest.fixture
def knowledge_pipeline(pipeline_context, pipelines_package):
    yaml_path = resources.files(pipelines_package).joinpath("knowledge.yaml")
    return Pipeline.from_file(pipeline_context, yaml_path)
