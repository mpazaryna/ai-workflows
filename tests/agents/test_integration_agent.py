import pytest

from agents.langchain_react import create_agent, create_llm, load_environment, run_query


def test_query_results():
    # Load environment variables
    env_vars = load_environment()

    # Create LLM and agent
    llm = create_llm(env_vars["OPENAI_API_KEY"])
    agent = create_agent(llm)

    # Run a simple query
    query = "What is the capital of France?"
    result = run_query(agent, query)

    # Assert that we got a result
    assert result is not None
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0

    # Print the result for inspection
    print(f"Query result: {result['output']}")
