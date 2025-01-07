# Standard
import unittest

# Third Party
import pytest

# First Party
from src.instructlab.sdg.datamixing import _get_question_hack, _get_response_hack
from src.instructlab.sdg.pipeline import SIMPLE_PIPELINES_PACKAGE


@pytest.fixture
def model():
    return "merlinite-7b-Q4_K_M.gguf"


@pytest.fixture
def model_family():
    return "merlinite"


@pytest.fixture
def model_repo_id():
    return "ibm/merlinite-7b-GGUF"


@pytest.fixture
def num_instructions_to_generate():
    return 2


@pytest.fixture
def pipelines_package():
    return SIMPLE_PIPELINES_PACKAGE


@pytest.mark.gpu
class TestSimplePipeline(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _setup_fixtures(self, knowledge_dataset, knowledge_pipeline):
        self.knowledge_dataset = knowledge_dataset
        self.knowledge_pipeline = knowledge_pipeline

    def test_knowledge(self):
        samples = self.knowledge_pipeline.generate(self.knowledge_dataset)
        assert len(samples) > 0
        for sample in samples:
            question = _get_question_hack(sample)
            response = _get_response_hack(sample)
            assert len(question) > 0
            assert len(response) > 0
