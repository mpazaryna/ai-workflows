import pytest

from pipelines.langchain_pipeline import (
    create_llm,
    create_pipeline,
    load_environment,
    run_pipeline,
)


def test_pipeline_result_exists():
    # Load environment variables
    env_vars = load_environment()

    # Create LLM
    llm = create_llm(env_vars)

    # Create the pipeline components
    google_search, math_chain, search_chain = create_pipeline(llm)

    # Run a simple query
    query = "How old was Vincent van Gogh when he died? What is his age raised to the 0.25 power?"
    result = run_pipeline(google_search, math_chain, search_chain, query)

    # Assert that we got a result
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

    # Print the result for inspection
    print(f"Pipeline result: {result}")
