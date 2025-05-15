# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from instructlab.sdg.utils import GenerateException, models


class TestModels:
    """Test model family in instructlab.sdg.utils.models."""

    def test_granite_model_family(self):
        assert (
            models.get_model_family("granite", "./models/granite-7b-lab-Q4_K_M.gguf")
            == "granite"
        )

    def test_merlinite_model_family(self):
        assert (
            models.get_model_family(
                "merlinite", "./models/merlinite-7b-lab-Q4_K_M.gguf"
            )
            == "merlinite"
        )

    def test_mixtral_model_family(self):
        assert (
            models.get_model_family(
                "mixtral", "./models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
            )
            == "mixtral"
        )

    def test_mistral_model_family(self):
        assert (
            models.get_model_family(
                "mistral", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            )
            == "mistral"
        )

    def test_default_model_family(self):
        assert (
            models.get_model_family(None, "./models/foo-8x7b-instruct-v0.1.Q4_K_M.gguf")
            == "merlinite"
        )
        assert (
            models.get_model_family("", "./models/foo-8x7b-instruct-v0.1.Q4_K_M.gguf")
            == "merlinite"
        )

    def test_model_family_overrides(self):
        assert (
            models.get_model_family(
                "mixtral", "./models/foo-8x7b-instruct-v0.1.Q4_K_M.gguf"
            )
            == "mixtral"
        )

    def test_unknown_model_family(self):
        with pytest.raises(GenerateException) as exc:
            models.get_model_family(
                "foobar", "./models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
            )
        assert "Unknown model family: foobar" in str(exc.value)

    def test_none_args(self):
        assert models.get_model_family(None, None) == "merlinite"
