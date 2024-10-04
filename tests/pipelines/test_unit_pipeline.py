import os
from unittest.mock import patch

import pytest

from pipelines.langchain_pipeline import load_environment


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
def test_load_environment_success():
    api_key = load_environment()
    assert api_key == "test_api_key"


@patch.dict(os.environ, {}, clear=True)
def test_load_environment_key_not_found():
    with pytest.raises(ValueError) as exc_info:
        load_environment()
    assert str(exc_info.value) == "OpenAI API key not found in environment variables."


if __name__ == "__main__":
    pytest.main([__file__])
