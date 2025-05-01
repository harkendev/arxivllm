from pydantic import BaseModel
from pydantic_ai import Agent
import logging
from textwrap import dedent

from .config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

class Query(BaseModel, use_attribute_docstrings=True):
    thought: str
    """ let's think step by step here """
    query: str

async def gen_query(user_query: str) -> Query:
    logger.debug(f"Generating ArXiv query for user input: '{user_query}'")
    agent = Agent(
        DEFAULT_MODEL,
        output_type=Query,
        instructions=dedent(
        """
        Generate an arXiv search query based on the user query.

        Here are the rules:
            - use fields qualifier for search terms:
                - ti: title
                - au: author
                - abs: abstract
                - cat: category
                - sub: sub-category
        """
        )
    )
    result = await agent.run(f"<UserQuery>\n{user_query}\n</UserQuery>")
    logger.debug(f"ArXiv query result: {result}")
    output_data: Query = result.output # type: ignore[assignment]

    return output_data