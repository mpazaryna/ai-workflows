import json
import logging
from pathlib import Path

# Import functions from langchain_react.py
from agents.langchain_react import create_agent, create_llm, load_environment
from agents.langchain_react import run_query as run_agent_query

# Import functions from langchain_pipeline.py
from pipelines.langchain_pipeline import create_pipeline
from pipelines.langchain_pipeline import run_pipeline as run_pipeline_query

# Configure logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "enhanced_hybrid_workflow.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="a",
)
logger = logging.getLogger(__name__)


def enhanced_hybrid_workflow(initial_query):
    try:
        logger.info(f"Starting enhanced hybrid workflow for query: {initial_query}")

        api_key = load_environment()
        llm = create_llm(api_key)

        # Step 1: Use the agent to analyze the query and determine sub-tasks
        agent = create_agent(llm)
        agent_result = run_agent_query(
            agent,
            f"Analyze this query and break it down into subtasks: {initial_query}. Return the result as a JSON string with 'subtasks' as the key and a list of subtasks as the value.",
        )

        # Parse the agent's response
        subtasks = json.loads(agent_result["output"])["subtasks"]
        logger.info(f"Subtasks identified: {subtasks}")

        # Step 2: Process each subtask using the pipeline
        google_search, math_chain, search_chain = create_pipeline(llm)
        subtask_results = []
        for subtask in subtasks:
            pipeline_result = run_pipeline_query(
                google_search, math_chain, search_chain, subtask
            )
            subtask_results.append({"task": subtask, "result": pipeline_result})

        # Step 3: Use the agent to synthesize the results
        synthesis_prompt = f"Given these subtask results: {json.dumps(subtask_results)}, provide a comprehensive answer to the original query: {initial_query}"
        final_result = run_agent_query(agent, synthesis_prompt)

        logger.info("Enhanced hybrid workflow completed successfully")
        return final_result["output"]

    except Exception as e:
        logger.exception(f"An error occurred in the enhanced hybrid workflow: {str(e)}")
        return f"An error occurred: {str(e)}"


def main():
    query = "Compare the lifespans of Vincent van Gogh and Pablo Picasso. Calculate the difference in their ages at death, and express this difference as a percentage of the average human lifespan today."
    result = enhanced_hybrid_workflow(query)
    print(result)


if __name__ == "__main__":
    main()
