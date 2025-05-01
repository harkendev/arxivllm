import typer
import logging
import asyncio

from arxivllm import find as find_arxiv_papers, present

app = typer.Typer()

def setup_logging(debug: bool = False):
    app_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=app_level, # Set default level based on debug flag
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING) 
    logging.getLogger("openai").setLevel(logging.WARNING) 
    logging.getLogger("urllib3").setLevel(logging.WARNING)

@app.command()
def find(
    user_query: str,
    debug: bool = False,
    num_results: int = 100
) -> None:
    """Find relevant arXiv papers based on your query and discuss them."""
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting find command with query: '{user_query}' for {num_results} results.")
    
    results = asyncio.run(find_arxiv_papers(user_query, nb_results=num_results))

    if results:
        logger.info(f"Passing {len(results)} sorted results to discussion UI.")
        present(results)
    else:
        logger.warning("No results were found or processed successfully.")
        typer.echo("Could not find or process any results.") 

    logger.info("Find command finished.")


if __name__ == "__main__":
    app()
