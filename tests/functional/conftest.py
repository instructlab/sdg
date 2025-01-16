# Standard
import pathlib
import typing

# Third Party
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
