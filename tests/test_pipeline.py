import os

import pytest

from pipelines.langchain_pipeline import (
    create_llm,
    create_pipeline,
    load_environment,
    main,
    run_pipeline,
)


@pytest.fixture(scope="module")
def api_key():
    return load_environment()


@pytest.fixture(scope="module")
def llm(api_key):
    return create_llm(api_key)


@pytest.fixture(scope="module")
def pipeline_components(llm):
    return create_pipeline(llm)


def test_load_environment():
    api_key = load_environment()
    assert api_key is not None
    assert len(api_key) > 0


def test_create_llm(api_key):
    llm = create_llm(api_key)
    assert llm is not None
    assert llm.model_name == "gpt-3.5-turbo-instruct"
    assert llm.temperature == 0


def test_create_pipeline(llm):
    google_search, math_chain, search_chain = create_pipeline(llm)
    assert google_search is not None
    assert math_chain is not None
    assert search_chain is not None


def test_run_pipeline(pipeline_components):
    google_search, math_chain, search_chain = pipeline_components
    query = "How old was Vincent van Gogh when he died? What is his age raised to the 0.25 power?"
    result = run_pipeline(google_search, math_chain, search_chain, query)

    assert "Vincent" in result
    # assert "raised to the 0.25 power" in result

    # Check if the result contains a number (age)
    assert any(char.isdigit() for char in result)


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Pipeline Result:" in captured.out
    # assert "Vincent van Gogh's age at death:" in captured.out
    # assert "raised to the 0.25 power" in captured.out
