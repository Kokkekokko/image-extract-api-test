# Image Extract API Test

A test harness that measures how accurately a vision-language AI API reads images.

It runs in four steps:

1. **Send.** Each image in `images/` goes to your API endpoint, with an optional per-image prompt.
2. **Store.** Every response is saved as JSON: one file per image plus a combined `responses.jsonl`.
3. **Compare.** Each response is scored against the correct answer with several metrics.
4. **Report.** An accuracy report is written as Markdown, HTML and JSON, with overall, per-task and per-category scores and the worst cases.

The harness is tested with [ThaiOCRBench](https://huggingface.co/datasets/typhoon-ai/ThaiOCRBench). You can use any dataset that comes as images plus a ground-truth file.

---

## Project layout

```
main.py                     CLI entry point: prepare | run | score | report | all
config.example.yaml         template; copy to config.yaml
.env.example                template for API keys; copy to .env
src/
  config.py                 YAML loading, ${ENV_VAR} expansion, .env support
  clients.py                API providers: openai_compatible, anthropic, http, echo
  dataset.py                reads the ground-truth file and pairs it with images
  runner.py                 steps 1+2: sends images, saves JSON (parallel, retries, resume)
  metrics.py                comparison metrics
  scorer.py                 step 3: writes scores.csv
  report.py                 step 4: writes report.md, report.html, summary.json
datasets_adapters/
  thaiocrbench.py           downloads a ThaiOCRBench sample into images/ + data/
images/                     input images
data/ground_truth.jsonl     correct answers
results/<run-name>/         output of each run
```

## Setup

You need Python 3.10 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp config.example.yaml config.yaml    # then edit config.yaml
cp .env.example .env                  # then add your API keys
```

Any `${NAME}` in `config.yaml` is replaced with the environment variable `NAME`, which can also be set in `.env`. Keep keys out of `config.yaml`. Both `config.yaml` and `.env` are git-ignored.

## Quick start with ThaiOCRBench

```bash
# 1. Download a small sample: 5 images per task type, 13 tasks, about 65 images
python main.py prepare thaiocrbench --per-task 5

# 2. Optional: check the pipeline offline. Set `api.provider: echo` in config.yaml.
#    Echo answers with the ground truth, so you should get 100%. Set `echo.mode: empty` to get 0%.
python main.py all --run-name echo_check

# 3. Run against the real API. Set api.provider and its section in config.yaml.
python main.py all --run-name foundry_gpt41
```

Open `results/foundry_gpt41/report.html` in a browser, or read `report.md`.

Other `prepare` options:

- `--per-task N` sets the number of images per task.
- `--limit N` caps the total number of images.
- `--all` downloads the full dataset: 2,808 images, about 1.9 GB.

To skip the Hugging Face rate-limit warning, set `HF_TOKEN`.

## Commands

| Command | What it does |
|---|---|
| `python main.py prepare thaiocrbench [--per-task N \| --limit N \| --all]` | Builds `images/` and `data/ground_truth.jsonl` |
| `python main.py run --run-name NAME [--limit N] [--rerun]` | Steps 1+2: sends the images and saves the responses |
| `python main.py score --run-name NAME` | Step 3: compares responses with the answers and writes `scores.csv` |
| `python main.py report --run-name NAME` | Step 4: builds the reports |
| `python main.py all --run-name NAME` | Runs steps 1 to 4 |

All commands accept `--config path.yaml`. The default is `config.yaml`.

**Resume.** Running the same `--run-name` again skips images that already have a saved response. Failed calls (network errors, 5xx) are retried on the next run. To send everything again, use `--rerun`.

**Re-scoring.** To try different metrics or normalization, change `metrics:` in the config and run `score` and `report` again. This does not call the API.

## Choosing an API provider

Set `api.provider` in `config.yaml` to one of the providers below.

### `openai_compatible`: Microsoft Foundry, Azure OpenAI, OpenAI, vLLM, Ollama

This provider uses the Chat Completions format, with the image sent as a base64 `data:` URL. Most vision models deployed on **Microsoft Foundry** accept this format, including GPT-4o/4.1, Phi-4-multimodal, Llama Vision and Mistral.

```yaml
api:
  provider: openai_compatible
  openai_compatible:
    base_url: ${FOUNDRY_BASE_URL}   # e.g. https://<resource>.openai.azure.com/openai/v1/
    api_key: ${FOUNDRY_API_KEY}
    model: gpt-4.1                  # your deployment name
    temperature: 0
```

Copy the endpoint URL and key from your deployment's page in the Foundry portal. If the endpoint uses a different URL shape, or needs an `api-version` query parameter, use the `http` provider instead.

### `anthropic`: Claude as a mock or reference model

Claude gives you a strong baseline to compare your own model against.

```yaml
api:
  provider: anthropic
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-opus-5-5          # claude-haiku-5-5 is a much cheaper mock
    effort: low
    foundry_resource: null          # set this to call Claude deployed on Microsoft Foundry
```

### `http`: any other REST API

You give the URL, the headers and a body template. The placeholders `{image_b64}`, `{mime}`, `{prompt}` and `{filename}` are filled in for each image. `response_path` tells the harness where the answer text sits in the JSON response.

```yaml
api:
  provider: http
  http:
    url: https://my-api.example.com/v1/extract
    headers: { Authorization: "Bearer ${MY_API_KEY}" }
    mode: json                      # or multipart (image sent as a file field)
    body: { image: "{image_b64}", prompt: "{prompt}" }
    response_path: result.text      # dotted path; numbers index lists, e.g. choices.0.message.content
```

### `echo`: offline pipeline test

This provider makes no network calls. `mode: answer` returns the correct answer and `mode: empty` returns nothing.

### Shared settings

| Setting | Meaning |
|---|---|
| `timeout` | Seconds before a request is abandoned |
| `max_retries` | Retries on 429, 5xx and timeouts, with exponential backoff |
| `concurrency` | Number of requests sent in parallel |
| `max_tokens` | Maximum length of the answer |

The text sent with each image is `prompt.system` followed by the image's `question`. If an image has no question, `prompt.default_question` is used instead.

## Using your own dataset

The harness needs two things:

1. Images in a folder. Set the folder in `dataset.images_dir`.
2. A ground-truth file, in `.jsonl`, `.csv` or `.json` (a list), with one row per image:

```json
{"id": "inv_001", "image": "inv_001.jpg", "question": "Extract the invoice total.", "answer": "1,250.00", "task": "KIE", "category": "Invoice"}
```

| Field | Required | Purpose |
|---|---|---|
| `image` | yes | File name inside `images_dir` |
| `answer` | yes, for scoring | The correct answer |
| `id` | no | Defaults to the image file name |
| `question` | no | Prompt for this image |
| `task`, `category` | no | Groupings used in the report |

If your columns have different names, map them in `dataset.fields`. You don't need to rename anything in the file:

```yaml
dataset:
  images_dir: my_images
  ground_truth: my_labels.csv
  fields: { id: doc_id, image: file, answer: label, question: prompt, task: type }
```

Images that have no ground-truth row are still sent and saved, but they are marked `unscored`. To collect responses without scoring, set `ground_truth: null`. To make a dataset reusable, add an adapter in `datasets_adapters/` modeled on `thaiocrbench.py`.

## How answers are compared

Before comparison, both texts are Unicode-normalized (NFC), Markdown code fences are removed and whitespace is collapsed. Optional normalization flags live under `metrics.normalize`.

| Metric | Meaning |
|---|---|
| `exact_match` | 1 if the normalized texts are identical, otherwise 0 |
| `edit_similarity` | 1 − Levenshtein distance / longer length, measured on characters |
| `anls` | `edit_similarity`, but 0 when it is below `anls_threshold` (default 0.5). Standard for VQA |
| `cer` | Character error rate: edits / answer length. **Lower is better**, and it can exceed 1 |
| `char_f1` | F1 over the multiset of characters, ignoring order and spaces |
| `token_f1` | F1 over whitespace-separated words. Weak for Thai, which doesn't put spaces between words |
| `json_f1` | For JSON answers: F1 over flattened `key → value` pairs. JSON is pulled out of the model's reply even when it is wrapped in text or a code fence |

String metrics work on characters, because Thai has no word boundaries.

Two answer formats get special handling:

- **Several accepted answers.** If the answer is a JSON list of strings, such as `["8", "อันดับ 8"]`, the best-matching one counts. ThaiOCRBench's VQA tasks use this format.
- **Structured answers.** JSON objects, such as key-information or chart answers, are also scored with `json_f1`.

Each item gets one **score** from its task's primary metric, which you set in `metrics.primary_by_task`. Other tasks use `default_primary`. **Overall accuracy** is the mean of those scores. Items with an API error and no text score 0.

Default primary metric for each ThaiOCRBench task:

| Task | Primary metric |
|---|---|
| Document classification | `exact_match` |
| Infographics, Cognition VQA, Diagram VQA | `anls` |
| Key information extraction, Key information mapping, Chart parsing | `json_f1` |
| OCR, handwriting, document and table parsing | `edit_similarity` |

Table answers are HTML. They are compared as text, which is a useful proxy but not a TEDS (tree-edit-distance) score. If you need numbers you can compare with the published benchmark, use the official ThaiOCRBench evaluation code: https://github.com/scb-10x/ThaiOCRBench.

## Output files

Each run writes these files to `results/<run-name>/`:

| File | Contents |
|---|---|
| `responses/<id>.json` | Prompt, response text, raw API response, latency, token usage and error for one image |
| `responses.jsonl` | All responses in one file, without the raw API payloads |
| `run_meta.json` | Provider, model, start and finish time |
| `scores.csv` | One row per image, with every metric plus expected and predicted text. Saved as UTF-8 with BOM so Excel shows Thai correctly |
| `summary.json` | Aggregated numbers, for comparing runs in code |
| `report.md` / `report.html` | Human-readable accuracy report |

## Notes

- **Cost.** Each image is one API call. Start with `--per-task 2` or `--limit 10` before a full run.
- **Windows and NumPy.** If `datasets` fails with *"pyarrow requires NumPy 2.0"*, your global Python has an older NumPy. Use the virtual environment from Setup.
- **ThaiOCRBench** is licensed under CC-BY-SA-4.0. Citation: *ThaiOCRBench*, arXiv:2511.04479 (IJCNLP-AACL 2025).
