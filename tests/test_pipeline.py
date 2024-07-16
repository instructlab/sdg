"""
Unit tests for common Pipeline functionality
"""

# Standard
from unittest import mock

# Third Party
import pytest

# First Party
from instructlab.sdg.pipeline import Pipeline

## Helpers ##


class CustomTypeError(TypeError):
    pass


class NoArgError(RuntimeError):
    """Exception that can't be instantiated with a single argument"""

    def __init__(self):
        super().__init__("no args")


@pytest.mark.parametrize(
    ["failure_exc", "exp_err_type"],
    [
        (CustomTypeError("Oh no!"), CustomTypeError),
        (NoArgError(), RuntimeError),
    ],
)
def test_pipeline_named_errors_match_type(failure_exc, exp_err_type):
    """Validate that block types and names appear in the error message from a
    pipeline exception and that the type of the error is preserved.
    """
    mock_dataset = ["not empty"]
    working_block = mock.MagicMock()
    working_block().generate.return_value = mock_dataset
    failure_block = mock.MagicMock()
    failure_block.__name__ = "BadBlock"
    failure_block().generate = mock.MagicMock(side_effect=failure_exc)
    pipe_cfg = [
        {"name": "I work", "type": "working", "config": {}},
        {"name": "I don't", "type": "failure", "config": {}},
    ]
    with mock.patch(
        "instructlab.sdg.pipeline._block_types",
        {
            "working": working_block,
            "failure": failure_block,
        },
    ):
        pipe = Pipeline(None, None, pipe_cfg)
        with pytest.raises(exp_err_type) as exc_ctx:
            pipe.generate(None)

        assert exc_ctx.value.__cause__ is failure_exc
        assert (
            str(exc_ctx.value)
            == f"BLOCK ERROR [{failure_block.__name__}/{pipe_cfg[1]['name']}]: {failure_exc}"
        )
