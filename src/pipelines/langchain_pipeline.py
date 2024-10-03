import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
log_file = log_dir / "langchain_pipeline.log"
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


def create_pipeline(llm):
    logger.info("Creating pipeline")
    google_search = GoogleSerperAPIWrapper()
    math_chain = LLMMathChain(llm=llm)

    search_prompt = PromptTemplate(
        input_variables=["query"],
        template="Search for information about {query} and provide a concise answer.",
    )
    search_chain = LLMChain(llm=llm, prompt=search_prompt)

    logger.info("Pipeline created successfully")
    return google_search, math_chain, search_chain


def run_pipeline(google_search, math_chain, search_chain, query):
    logger.info(f"Running pipeline for query: {query}")
    with get_openai_callback() as cb:
        # Step 1: Search for Van Gogh's age
        search_result = google_search.run(f"Vincent van Gogh age at death")

        # Step 2: Extract the age using LLM
        age_query = f"How old was {query} when he died?"
        age_result = search_chain.run(age_query)

        # Add this part to process the math query
        age = int(re.search(r"\d+", age_result).group())
        math_query = f"What is {age} raised to the 0.25 power?"
        math_result = math_chain.run(math_query)

        logger.info(f"Total Tokens: {cb.total_tokens}")
        logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
        logger.info(f"Completion Tokens: {cb.completion_tokens}")
        logger.info(f"Total Cost (USD): ${cb.total_cost}")

    return f"{age_result}\nAnswer: {math_result}"


def main():
    try:
        api_key = load_environment()
        llm = create_llm(api_key)
        google_search, math_chain, search_chain = create_pipeline(llm)

        query = "How old was Vincent van Gogh when he died? What is his age raised to the 0.25 power?"
        result = run_pipeline(google_search, math_chain, search_chain, query)

        logger.info("Pipeline Result:")
        logger.info(result)

        print("Pipeline Result:")
        print(result)
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")


if __name__ == "__main__":
    main()
