import pytest

from agents.langchain_react import create_agent, create_llm, load_environment, run_query


@pytest.fixture(scope="module")
def api_key():
    return load_environment()


@pytest.fixture(scope="module")
def llm(api_key):
    return create_llm(api_key)


@pytest.fixture(scope="module")
def agent(llm):
    return create_agent(llm)


def test_run_query_returns_data(agent):
    # Arrange
    query = "What is the capital of France?"

    # Act
    result = run_query(agent, query)

    # Assert
    assert result is not None
    assert isinstance(result, dict)
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0
    assert "Paris" in result["output"]


@pytest.mark.skip(reason="Skipping this test for now")
def test_run_query_handles_complex_query(agent):
    # Arrange
    query = "How old was Vincent van Gogh when he died? What is his age raised to the 0.25 power?"

    # Act
    result = run_query(agent, query)

    # Assert
    assert result is not None
    assert isinstance(result, dict)
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0
    assert "37" in result["output"]  # Van Gogh's age at death
    assert any(num in result["output"] for num in ["2.53", "2.54"])  # 37^0.25 ≈ 2.53722
