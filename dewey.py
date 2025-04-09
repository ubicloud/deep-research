import argparse
import io
import os
import re
import json
import functools
import logging
from enum import Enum
from datetime import datetime
from typing import Any, Dict, Optional, List, TypedDict

import requests
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from tavily import TavilyClient
from markdown_pdf import MarkdownPdf, Section
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load UBICLOUD_API_KEY from .env file
load_dotenv()
UBICLOUD_API_KEY = os.getenv("UBICLOUD_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model configurations
SUMMARIZATION_MODEL = "mistral-small-3"
SUMMARIZATION_CONTENT_CUTOFF = 50000
REASONING_MODEL = "ds-r1-qwen-32b"
WRITING_MODEL = "mistral-small-3"
JSON_MODEL = "ds-r1-qwen-32b"


class InferenceMode(Enum):
    """Enum for different inference modes."""
    SUMMARIZATION = 1
    REASONING = 2
    WRITING = 3
    JSON = 4

    def get_model(self) -> str:
        """Retrieve the corresponding model name for the inference mode."""
        return globals().get(f"{self.name}_MODEL")


@functools.lru_cache(maxsize=128)
def get_search_client(search_engine: str):
    if search_engine == "duckduckgo":
        return DDGS()
    elif search_engine == "tavily":
        return TavilyClient(TAVILY_API_KEY)
    raise Exception("Unsupported search engine")


@functools.lru_cache(maxsize=128)
def get_inference_client(model: str) -> openai.OpenAI:
    """Get the inference client for the specified model."""
    base_url = f"https://{model}.ai.ubicloud.com/v1/"
    return openai.OpenAI(api_key=UBICLOUD_API_KEY, base_url=base_url)


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Look for code block (```...```) and parse as JSON."""
    match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("JSON decoding failed: %s", e)
            return None
    return None


def inference(messages: List[Dict[str, str]], mode: InferenceMode) -> Any:
    """Perform inference using the specified messages and inference mode."""
    model = mode.get_model()
    inference_client = get_inference_client(model)
    params = {
        "model": model,
        "messages": messages,
    }
    logger.debug(json.dumps(params, indent=2))
    completion = inference_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = completion.choices[0].message.content
    logger.debug(content)
    # Remove everything before the </think> from the content
    content = re.sub(r'.*?</think>', '', content, flags=re.DOTALL).strip()
    if mode == InferenceMode.JSON:
        json_content = extract_json(content)
        if not json_content:
            logger.warning("No JSON object found. Retrying...")
            return inference(messages, mode)
        return json_content
    return content


def create_system_message(content: str) -> Dict[str, str]:
    """Create a structured system message for chat completions."""
    return {"role": "system", "content": content}


def create_user_message(content: Any) -> Dict[str, str]:
    """Create a structured user message for chat completions."""
    if isinstance(content, dict):
        # Filter out keys that start with '_'
        content_str = json.dumps({
            key: value for key, value in content.items() if not key.startswith("_")
        }, indent=2)
    else:
        content_str = str(content)
    return {"role": "user", "content": content_str}


@functools.lru_cache(maxsize=128)
def search(query: str, search_engine: str) -> List[Dict[str, Any]]:
    """Search for a query on the web and return the top results."""
    truncated_query = query[:200]
    logger.info(f"Searching for \"{truncated_query}\"")
    client = get_search_client(search_engine)
    if search_engine == "duckduckgo":
        search_results = client.text(truncated_query, max_results=5)
    elif search_engine == "tavily":
        search_results = client.search(query=truncated_query)["results"]
    logger.debug(json.dumps(search_results, indent=2))
    return search_results


def read_pdf(pdf_file: io.BytesIO) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)


@functools.lru_cache(maxsize=128)
def fetch_url(url: str) -> Optional[str]:
    """Fetch content from a URL."""
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            logger.error(
                f"Error fetching URL: {url} returned {response.status_code}")
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            pdf_file = io.BytesIO(response.content)
            return read_pdf(pdf_file)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logger.error(f"Error fetching URL: {url}. Exception: {e}")
        return None


class ResearchState(TypedDict):
    """Represents the research state."""
    topic: str
    search_results: List[Dict[str, Any]]
    outline: str
    report: str
    _visited: List[str]


@functools.lru_cache(maxsize=128)
def summarize(title: str, content: str) -> Optional[str]:
    """Summarize the given content based on the provided title."""
    try:
        return inference([
            create_system_message("""Summarize the content.
Only include information that is relevant to the title.
Be succinct. Summarize directly, don't repeat the title."""),
            create_user_message({
                "title": title,
                "content": content[:SUMMARIZATION_CONTENT_CUTOFF]
            })
        ], mode=InferenceMode.SUMMARIZATION)
    except Exception as e:
        logger.error(f"Error summarizing title '{title}': {e}")
        return None


def fetch_and_summarize(topic: str, search_result: Dict[str, Any]) -> Optional[str]:
    """Fetch and summarize content from a search result URL."""
    title = search_result.get("title", "No Title")
    url = search_result.get("url")
    logger.info(
        f"Reading \"{title}\" with {InferenceMode.SUMMARIZATION.get_model()}")
    content = fetch_url(url)
    if not content:
        logger.warning(f"No content fetched from URL: {url}")
        return None
    return summarize(title, content)


def gather_information(topic: str, state: ResearchState, search_engine: str) -> ResearchState:
    """Gather and update research information about a topic."""
    logger.info(f"Gathering information about \"{topic}\"")
    raw_search_results = search(topic, search_engine)
    search_results: List[Dict[str, Any]] = state.get("search_results", [])
    visited: List[str] = state.get("_visited", [])

    for raw_search_result in raw_search_results:
        # Some search engine stores the url as href. We copy href to url.
        url = raw_search_result.get("href") or raw_search_result.get("url")
        raw_search_result["url"] = url
        if url in visited:
            continue
        visited.append(url)
        summary = fetch_and_summarize(topic, raw_search_result)
        if summary:
            search_results.append({
                "title": raw_search_result.get("title", "No Title"),
                "url": url,
                "summary": summary,
                "index": len(search_results) + 1
            })
    logger.debug(str(fetch_url.cache_info()))
    updated_state: ResearchState = {
        **state,
        "search_results": search_results,
        "_visited": visited
    }
    return updated_state


def thinking(state: ResearchState) -> ResearchState:
    """Think about the topic."""
    logger.info(
        f"Thinking about \"{state['topic']}\" with {InferenceMode.REASONING.get_model()}"
    )
    # Remove any existing reasoning keys to ensure a fresh response
    state.pop("thinking", None)
    state.pop("outline", None)
    state.pop("report", None)
    thinking = inference([
        create_system_message("""
Take a deep breath, go through the search results, think about the topic step by step, and provide a well-reasoned answer.
Use information from the search results as needed. Ignore unrelated search results.
If data is unavailable, analyze the problem step by step using existing information. Make reasonable assumptions, develop well-informed estimates, and cross-verify results through multiple approaches whenever possible.
Be objective, reasonable, and comprehensive."""),
        create_user_message(state)
    ], mode=InferenceMode.REASONING)
    return {**state, "thinking": thinking}


def deep_dive(state: ResearchState, search_engine: str) -> ResearchState:
    """Deep dive into a few areas."""
    subtopics = inference([
        create_system_message("""
Identify 3 key areas for deeper exploration on the given topic and thinking.
Return a JSON array of strings.
Each string should be a well-structured search engine query."""),
        create_user_message({
            "topic": state["topic"],
            "thinking": state.get("thinking", "")
        })
    ], InferenceMode.JSON)
    if subtopics is None:
        logger.warning("No JSON object found. Retrying deep_dive...")
        return deep_dive(state, search_engine)
    for subtopic in subtopics:
        state = gather_information(subtopic, state, search_engine)
    return state


def create_outline(state: ResearchState) -> ResearchState:
    """Create an outline of the report."""
    logger.info(
        f"Creating an outline of the report on \"{state['topic']}\" with {InferenceMode.REASONING.get_model()}"
    )
    state.pop("outline", None)
    state.pop("report", None)
    outline = inference([
        create_system_message("""
Generate an outline of a professional report on the given topic and thinking.
Think step by step. Use search results as needed. Ignore unrelated search results."""),
        create_user_message(state)
    ], InferenceMode.REASONING)
    return {**state, "outline": outline}


def write_report(state: ResearchState) -> ResearchState:
    """Write the report."""
    logger.info(
        f"Writing a report on \"{state['topic']}\" with {InferenceMode.WRITING.get_model()}"
    )
    state.pop("report", None)
    report = inference([
        create_system_message(
            """Generate an extremely detailed professional report based on the provided topic, thinking, and outline.
Use search results as needed. Ignore unrelated search results.
Ensure it is well-organized and each section is well-developed.
Use subsections and lists as needed.
Include the topic as the title. Include an executive summary at the beginning.
Refer to search results by their index as needed. Do not include the list of references at the end.
Use heading level 1 for the title.
Do not include figures."""),
        create_user_message(state)
    ], InferenceMode.WRITING)
    references = [
        f"1. {result['title']}. (n.d.). Retrieved from {result['url']}"
        for result in state["search_results"]]
    report += "\n\n## References\n\n" + \
        "\n".join(references) + "\n\nUbicloud AI"
    return {**state, "report": report}


CUSTOM_MARKDOWN_PDF_CSS = """
body { font-family: sans-serif; line-height: 1.5; margin: 1em; color: #333; }
h1, h2, h3 { color: #2c3e50; margin: 1em 0 0.5em; }
h1 { font-size: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
p { margin: 0.8em 0; }
code, pre code { font-family: monospace; background: #f4f4f4; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.8em; color: #333; }
pre code { color: #333; padding: 1em; white-space: pre-wrap; word-break: break-word; }
blockquote { border-left: 4px solid #ddd; padding-left: 1em; color: #666; margin: 1em 0; }
ul, ol { margin: 0.8em 0; padding-left: 1.2em; }
table { width: 100%; border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 0.5em; }
th { background: #f9f9f9; }
"""


def save_pdf(topic: str, report: str, filename: str) -> None:
    """Save the generated report as a PDF file with a custom CSS style."""
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(report), user_css=CUSTOM_MARKDOWN_PDF_CSS)
    full_filename = f"{filename}.pdf"
    pdf.save(full_filename)
    logger.info(f"Saved PDF: {full_filename}")


def save_state_json(topic: str, state: dict, filename: str) -> None:
    """Save the research state as a JSON file."""
    full_filename = f"{filename}.json"
    with open(full_filename, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved state JSON: {full_filename}")


def get_filename(topic: str) -> str:
    """Generate a filename based on the topic and timestamp."""
    logger.info(
        f"Creating a filename of the report with {InferenceMode.JSON.get_model()}"
    )
    basename = inference([
        create_system_message("""
Create a concise file name of lower case alphanumeric characters and underscore for the given topic.
Do not include extension. Return as a raw json string. Do not include any fields. Do include triple backtick."""),
        create_user_message(topic)
    ], InferenceMode.JSON)
    if not re.fullmatch(r'[a-z0-9_]+', basename):
        return get_filename(topic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{basename}_{timestamp}"


def deep_research(topic: str, depth: int, search_engine: str, initial_state: Optional[dict]) -> None:
    """Conduct deep research on a given topic and generate a PDF report"""
    filename = get_filename(topic)
    state = initial_state or {}
    state["topic"] = topic
    state["search_results"] = []
    save_state_json(topic, state, filename)
    state = gather_information(topic, state, search_engine)
    state = thinking(state)
    save_state_json(topic, state, filename)
    for _ in range(depth):
        state = deep_dive(state, search_engine)
        state = thinking(state)
        save_state_json(topic, state, filename)
    state = create_outline(state)
    state = write_report(state)
    save_pdf(topic, state["report"], filename)
    save_state_json(topic, state, filename)


def main() -> None:
    """Parse command-line arguments and start the deep research process."""
    parser = argparse.ArgumentParser(
        description="Perform deep research on a given topic."
    )
    parser.add_argument("topic", type=str, help="The topic to research")
    parser.add_argument("--depth", type=int, default=3,
                        help="The depth of the research")
    parser.add_argument("--resume", type=str,
                        help="Path to a JSON file to resume state from")
    parser.add_argument("--search_engine", type=str, default="duckduckgo",
                        help="The search engine to use, either DuckDuckGo or Tavily.")
    args = parser.parse_args()

    initial_state: Optional[dict] = None
    if args.resume:
        try:
            with open(args.resume, "r", encoding="utf-8") as f:
                initial_state = json.load(f)
            logger.info(f"Resumed state loaded from {args.resume}")
        except Exception as e:
            logger.error(
                f"Failed to load resume state from {args.resume}: {e}")

    deep_research(args.topic, args.depth, args.search_engine, initial_state)


if __name__ == "__main__":
    main()
