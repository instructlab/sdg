# Standard
from importlib import resources
import unittest

# Third Party
import pytest

# First Party
from src.instructlab.sdg.datamixing import _get_question_hack, _get_response_hack
from src.instructlab.sdg.pipeline import (
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)

# Local
from .llama_cpp_helpers import llama_cpp_openai_client


@pytest.mark.gpu
class TestSimplePipeline(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _setup_fixtures(self, tonsils_knowledge_dataset):
        model = "merlinite-7b-Q4_K_M.gguf"
        model_repo_id = "ibm/merlinite-7b-GGUF"
        model_family = "merlinite"
        client = llama_cpp_openai_client(model, model_repo_id)
        teacher_model = client.models.list().data[0].id
        num_instructions_to_generate = 2
        max_num_tokens = 1024
        context = PipelineContext(
            client=client,
            model_family=model_family,
            model_id=teacher_model,
            num_instructions_to_generate=num_instructions_to_generate,
            max_num_tokens=max_num_tokens,
        )
        yaml_path = resources.files(SIMPLE_PIPELINES_PACKAGE).joinpath("knowledge.yaml")
        self.knowledge_dataset = tonsils_knowledge_dataset
        self.knowledge_pipeline = Pipeline.from_file(context, yaml_path)

    def test_knowledge(self):
        samples = self.knowledge_pipeline.generate(self.knowledge_dataset)
        assert len(samples) > 0
        assert "output" in samples.column_names
        for sample in samples:
            question = _get_question_hack(sample)
            response = _get_response_hack(sample)
            assert len(question) > 0
            assert len(response) > 0
