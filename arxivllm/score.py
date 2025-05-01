from pydantic_ai import Agent
from pydantic import BaseModel
import logging
from typing import Tuple
from arxiv import Result  # type: ignore

from .config import DEFAULT_MODEL

from textwrap import dedent

logger = logging.getLogger(__name__)


class Score(BaseModel, use_attribute_docstrings=True):
    deliberation: str
    """ What you consider when scoring a result """
    justification: str
    """ The justification or explanation as to why you gave the score """
    score: float
    """ The numerical value of the score between 0 and 100 (more is better) """


async def score(result: Result, query: str) -> Tuple[Score, Result]:
    logger.debug(f"Scoring result '{result.entry_id}' against query '{query}'")
    agent = Agent(
        DEFAULT_MODEL,
        output_type=Score,
        instructions=dedent(
            """
            You are an agent whose job is to score results of a scientific
            paper search query based on a given user initial query.

            You will judge the results based on the following criteria:

            1. The relevance of the result with respect to the user query.
            2. The quality of the result in terms of clarity and precision.
            3. The reputation of the authors.

            You will output a score from 0-100.

            Examples
            --------
            user_query: >
                I am looking for some proven prompt engineering techniques to 
                improve the results of LLM call for my AI agent I am building.
            title: Prompt Engineering Techniques
            summary: This article reviews the state of the art in prompt 
            engineering with an emphasis on building reliable AI agents. 
            We will cover some of the most oftenly used techniques as well as 
            how they improce the results of LLM calls.
            authors: Yann LeCun, Yoshua Bengio, Geoffrey Hinton
            {
                "knowledge": "prompt engineering is the study of the methods
                used to create effective prompts for Large Language Models
                (LLMs). A prompt is an input given to an LLM to guide it's
                output. The query emphasize the need for proven prompt 
                engineering techniques, emphasizing the need to integrate in
                the development of AI agents. The article summary mentions that
                is is a review article, which seems perfect since the user is 
                looking for proven techniques. It also seems to be general
                and this is what we want because the user did not mention 
                the specific application domain.",
                "justification": "The article is a review article by extremely
                renowned authors. It is a general article that can be applied
                to any application domain. The abstract mentions that
                it is about building reliable AI agents, which is what the
                user is looking for.",
                "score": 100
            }
            """
        ),
    )

    prompt = dedent(
        f"""
        user_query: >
            {query}
        title: {result.title}
        summary: {result.summary}
        authors: {",".join((author.name for author in result.authors))}
        """
    )

    try:
        response = await agent.run(prompt)
        score_output: Score = response.output  # type: ignore[assignment]
        return score_output, result
    except Exception:
        logger.exception(f"Failed to score result '{result.entry_id}'")
        return Score(
            deliberation="Failed to score due to error.", justification="", score=-1.0
        ), result
