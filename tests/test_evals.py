# create evals for the arxivllm tool
from textwrap import dedent

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from arxivllm.config import DEFAULT_JUDGE_MODEL
from arxivllm.query import gen_query

# create the data set of the different user queries.
import pytest

queries = [
    Case(
        name="find-few-shot-learning",
        inputs=dedent(
        """
        looking for the paper that really kicked off the modern era of 
        large language models. Like, where scaling up huge models first 
        really showed surprising generalization, few-shot, and zero-shot 
        capabilities without explicit retraining. 
        """),
        metadata={"function": "query"},
    ),
    Case(
        name="find-auto-ml",
        inputs=dedent(
        """
        I'm trying to get a broad overview of AutoML techniques. I want to 
        know what components of the machine learning pipeline can be 
        automated, and what kinds of algorithms or strategies are used to 
        search over model architectures or hyperparameters. Ideally, the 
        paper gives me a mental map of the AutoML field. Kind of like a 
        cheat sheet of approaches and their pros and cons.
        """
        ),
        metadata={"function": "query"},
    ),
    Case(
        name="find-transformer",
        inputs=dedent(
        """
        I'm trying to understand how modern language models like GPT and 
        BERT work under the hood. I want a paper that introduces the 
        architecture behind these systems—especially something that moved 
        away from traditional RNNs and LSTMs. Ideally, it explains the 
        fundamental building blocks of today's transformer models and why 
        that shift was important.
        """
        ),
        metadata={"function": "query"},
    ),
    Case(
        name="find-parameter-efficient-finetuning",
        inputs=dedent(
         """
        I'm looking for papers that explain techniques for fine-tuning large 
        language models without needing to update all their parameters. Methods 
        like adapters, LoRA, or prefix-tuning that make it cheaper to adapt 
        giant models to new tasks. I want a paper that gives a clear overview 
        of these parameter-efficient fine-tuning strategies.
        """
        ),
        metadata={"function": "query"},
    ),
    Case(
        name="find-llm-memory-architectures",
        inputs=dedent(
        """
        I want to find papers about giving large language models some kind of 
        persistent or dynamic memory. Like, architectures where the model can 
        store information outside the attention window and retrieve it later. 
        Ideally, something about memory-augmented transformers or agent memory 
        systems that improve long-term reasoning.
        """
        ),
        metadata={"function": "query"},
    ),
    Case(
        name="find-evaluation-robustness",
        inputs=dedent(
        """
        I'm trying to find papers that discuss how we evaluate large language 
        models for robustness. Not just accuracy benchmarks, but stress tests 
        that reveal brittleness, distribution shifts, adversarial examples, or 
        hallucination rates. I'd like something that surveys or proposes 
        rigorous evaluation frameworks beyond standard leaderboards.
        """
        ),
        metadata={"function": "query"},
    ),
]

# Verify the efficacy of the LLM query generator.
query_dataset = Dataset(
    cases=queries,
    evaluators=[
        LLMJudge(
            model=DEFAULT_JUDGE_MODEL,  # type: ignore
            rubric=dedent(
                """
            The query uses the correct syntax for a search query on Arxiv.

            It must:
            - specify the appropriate fields in the query:
                - ti: for titles
                - au: for authors
                - abs: for abstracts
                - cat: for categories
                - jr: for journal references
                - co: for comments
            - combine it's multiple search terms using the correct logical operators:
                - AND, OR, ANDNOT
            - use parentheses to group logical expressions
            - use quotes for exact phrases (e.g: "prompt engineering")
            - use the correct syntax for categories:
                - e.g. cat:cs.AI
            
            Valid query examples:
            (ti:planning OR ti:"task planning") AND (cat:cs.AI OR cat:cs.CL)
            (ti:"attention is all you need") AND (cat:cs.CV OR cat:cs.NE)
            """
            ),
            include_input=True,
        ),
    ],
)


@pytest.mark.asyncio
async def test_query_accuracy():
    """
    Must generate accurate queries for more then 90% of test cases.
    """
    report = await query_dataset.evaluate(gen_query)
    accuracy = report.averages().assertions

    assert accuracy >= 0.9


def print_report():
    report = query_dataset.evaluate_sync(gen_query)
    print(report)


if __name__ == "__main__":
    print_report()
