# Standard
import unittest

# Third Party
import pytest

# First Party
from src.instructlab.sdg.datamixing import _get_question_hack, _get_response_hack
from src.instructlab.sdg.pipeline import FULL_PIPELINES_PACKAGE


@pytest.fixture
def model():
    return "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    # return "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    # return "mistral-7b-instruct-v0.2.Q3_K_S.gguf"


@pytest.fixture
def model_family():
    return "mixtral"


@pytest.fixture
def model_repo_id():
    return "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"


@pytest.fixture
def num_instructions_to_generate():
    return 1


@pytest.fixture
def pipelines_package():
    return FULL_PIPELINES_PACKAGE


@pytest.mark.slow
class TestFullPipeline(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _setup_fixtures(self, knowledge_dataset, knowledge_pipeline):
        self.knowledge_dataset = knowledge_dataset
        self.knowledge_pipeline = knowledge_pipeline

    def test_knowledge(self):
        samples = self.knowledge_pipeline.generate(self.knowledge_dataset)
        print(samples)
        assert len(samples) > 0
        for sample in samples:
            print(sample)
            question = _get_question_hack(sample)
            response = _get_response_hack(sample)
            assert len(question) > 0
            assert len(response) > 0
