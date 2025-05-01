# arxivllm

Helps you find papers on arXiv.org.

## Requirements

You need to have an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Installation

```bash
# clone and cd into the repo
git clone https://github.com/harkendev/arxivllm.git
cd arxivllm

# create and activate the virtual environment
uv venv
source venv/bin/activate 

# install the package
uv pip install . 
```

## Usage

```bash
# Basic usage
arxivllm "your research query here"

# Example
arxivllm "recent papers about transformer architecture in computer vision"
```

## Features
1. Helps you find papers on arXiv.org using natural language queries.
2. Uses LLMs to score the search results by relevance.
3. Presents the results in order of best fit and provides a justification for the score.
4. Download selected papers locally.

## License
[MIT License](https://choosealicense.com/licenses/mit/)

## Author
Jonathan Pelletier.


