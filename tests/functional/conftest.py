# Standard
import pathlib
import typing

# Third Party
from datasets import Dataset
import pytest

TESTS_PATH = pathlib.Path(__file__).parent.parent.absolute()
EXAMPLES_PATH = TESTS_PATH.parent.joinpath("docs", "examples")


@pytest.fixture
def testdata_path() -> typing.Generator[pathlib.Path, None, None]:
    """Path to local test data directory"""
    yield TESTS_PATH / "testdata"


@pytest.fixture
def examples_path() -> typing.Generator[pathlib.Path, None, None]:
    """Path to examples directory"""
    yield EXAMPLES_PATH


@pytest.fixture
def tonsils_knowledge_dataset():
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
