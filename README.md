# Dewey.py: Rebuilding Deep Research with Open Models

Deep Research is an AI-powered solution that automates multi-step research processes, mimicking the work of a seasoned research analyst.
In this repo, we demonstrate how to build Dewey, your own deep research agent, using only open source.
This way, you can see how it works under the hook, customize the process based on your own needs, and deploy it on your own infrastructure.

To give you a sense of the style and quality of Dewey, here are a few sample reports it has generated:

- Technology: [Differences between Microsoft’s and Google’s recent quantum breakthroughs.](https://drive.google.com/file/d/1Pjvcv-I9xhdmnN-qf9PzFaWPxh3m8C9b/view?usp=sharing)
- Healthcare: [Advances in gene therapy for Alzheimer’s disease.](https://drive.google.com/file/d/1BS9Z2WchFwqXF40Rb7hubZrz-okeaqrD/view?usp=sharing)
- Travel: [A two-day itinerary for exploring New York City.](https://drive.google.com/file/d/1YuJeHm5VYmzD8tXWxULgZO8m4reYILwR/view?usp=share_link)

Each research takes about 5 to 10 minutes to run and costs less than 10 cents.

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```
UBICLOUD_API_KEY=your_api_key_here
```

You can obtain an API key by signing up on [Ubicloud](https://www.ubicloud.com) and generating one from the "AI Inference" page.

## Usage

Run research:

```bash
python dewey.py "Artificial Intelligence"
```

Set research depth (default is 3):

```bash
python dewey.py "Quantum Computing" --depth=2
```

Resume from saved state:

```bash
python dewey.py "Artificial Intelligence" --resume="saved_state.json"
```

By default, we use DuckDuckGo as the search engine, but it may encounter rate limits.
To avoid this, you can switch to Tavily search.
To do so, register at [tavily.com](https://tavily.com) to obtain an API key, then add `TAVILY_API_KEY=your_api_key_here` to the `.env` file, and run

```bash
python dewey.py "Artificial Intelligence" --search_engine="tavily"
```

## Output

- **PDF Report**: Structured report with summary and references.
- **JSON State**: Save progress for resuming research.

## License

MIT License.

🚀 Happy researching!