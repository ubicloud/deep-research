# Deep Research

A Python implementation of deep research.

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```
UBICLOUD_API_KEY=your_api_key_here
```

## Usage

Run research:

```bash
python deep_research.py "Artificial Intelligence"
```

Set research depth:

```bash
python deep_research.py "Quantum Computing" --depth=2
```

Resume from saved state:

```bash
python deep_research.py "Artificial Intelligence" --resume="saved_state.json"
```

## Output

- **PDF Report**: Structured report with summary and references.
- **JSON State**: Save progress for resuming research.

## License

MIT License.

🚀 Happy researching!