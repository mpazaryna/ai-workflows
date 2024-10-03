import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent 

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms import OpenAI

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
log_file = log_dir / "langchain_react.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="a",
)
logger = logging.getLogger(__name__)


def load_environment():
    logger.info("Loading environment variables")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        raise ValueError("OpenAI API key not found in environment variables")
    logger.info("Environment variables loaded successfully")
    return api_key


def create_llm(api_key):
    logger.info("Creating LLM instance")
    return OpenAI(api_key=api_key, model_name="gpt-3.5-turbo-instruct", temperature=0)


def create_agent(llm):
    logger.info("Creating agent")
    tools = load_tools(["google-serper", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    logger.info("Agent created successfully")
    return agent


def run_query(agent, query):
    logger.info(f"Running query: {query}")
    with get_openai_callback() as cb:
        result = agent.invoke(query)
        logger.info(f"Total Tokens: {cb.total_tokens}")
        logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
        logger.info(f"Completion Tokens: {cb.completion_tokens}")
        logger.info(f"Total Cost (USD): ${cb.total_cost}")
    return result


def main():
    try:
        api_key = load_environment()
        llm = create_llm(api_key)
        agent = create_agent(llm)

        query = "How old was Vincent van Gogh when he died? What is his age raised to the 0.25 power?"
        result = run_query(agent, query)

        logger.info("Query Result:")
        logger.info(result)

        print("Query Result:")
        print(result)
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")


if __name__ == "__main__":
    main()
