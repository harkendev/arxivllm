import asyncio
import logging
import textwrap
from typing import List, Tuple

import arxiv  # type: ignore
import typer

from .config import ARTICLE_DIR, TEXT_WRAP_WIDTH
from .query import gen_query
from .score import Score, score

from pathlib import Path    

logger = logging.getLogger(__name__)
ScoredResult = Tuple[Score, arxiv.Result]

async def find(user_query: str, nb_results: int = 100) -> List[ScoredResult]:
    try:
        arxiv_query = await gen_query(user_query)
    except Exception:
        logger.warning("Failed to generate ArXiv query from LLM.")
        return []

    logger.debug(f"ArXiv query: {arxiv_query.query}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=arxiv_query.query,
        max_results=nb_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results_iterator = client.results(search)
    score_tasks = [score(result, user_query) for result in results_iterator]

    processed_results = await asyncio.gather(*score_tasks)

    processed_results.sort(key=lambda x: x[0].score, reverse=True)
    return processed_results


def present(results: List[ScoredResult]):
    total_results = len(results)
    current_index = 0
    wrap_width = TEXT_WRAP_WIDTH

    while current_index < total_results:
        score_val, result = results[current_index]

        typer.echo(f"\n--- Result {current_index + 1} / {total_results} ---")
        typer.echo(f"Score: {score_val.score:.2f}")
        typer.echo(f"Title: {result.title}")
        typer.echo(f"Authors: {', '.join(str(a) for a in result.authors)}")

        summary = result.summary if result.summary else "No summary available."
        summary_lines = textwrap.fill(
            summary, 
            width=wrap_width, 
            initial_indent="    ", subsequent_indent="    "
        )
        typer.echo(f"Summary:\n{summary_lines}")

        justification = getattr(score_val, "justification", "N/A")
        justification_lines = textwrap.fill(
            justification,
            width=wrap_width,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        typer.echo(f"Justification:\n{justification_lines}")
        typer.echo("-" * wrap_width)

        # --- Prompt for Action ---
        prompt = "Actions: [D]ownload, [N]ext Result, [Q]uit: "
        action = input(prompt).strip().lower()

        path = Path(ARTICLE_DIR)
        path.mkdir(parents=True, exist_ok=True)

        match action:
            case "d":
                result.download_pdf(ARTICLE_DIR)
                current_index += 1
            case "n":
                current_index += 1
            case "q":
                break
            case _:
                typer.echo("Invalid action. Please choose D, N, or Q.")

        # Final message if loop finishes without quitting early
        if current_index >= total_results and action != "q":
            typer.echo("Finished reviewing all results.")
