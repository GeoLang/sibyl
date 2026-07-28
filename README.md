# sibyl

Generic LLM agent loop as a microservice: it calls an OpenAI-compatible chat completions endpoint, dispatches tool calls over HTTP to an external executor, stores session history in sqlite, and streams run events as NDJSON.

## Env

| var | default | notes |
| --- | --- | --- |
| `XAI_API_KEY` | required on the default base | see below |
| `SIBYL_HOST` | `0.0.0.0` | |
| `SIBYL_PORT` | `8090` | |
| `SIBYL_DB_PATH` | `/data/sibyl.db` | parent dir is created |
| `SIBYL_MODEL` | `grok-4-1-fast-reasoning` | |
| `SIBYL_API_BASE` | `https://api.x.ai/v1` | trailing slash is stripped |
| `GEOLANG_URL` | `http://geolang-api:8080` | tool manifest and executor |
| `SIBYL_TOOL_TIMEOUT_SECS` | `600` | per tool call |

An empty value counts as unset and falls back to the default, so a compose `${VAR:-}` pass-through cannot blank the base URL or the model.

`XAI_API_KEY` is required while `SIBYL_API_BASE` is the default x.ai endpoint, and startup fails without it. Point `SIBYL_API_BASE` at another server and the key becomes optional: with no key sibyl sends no `Authorization` header and logs one line saying so. If you do set a key alongside a custom base, it goes to that host, so only point sibyl at servers you trust.

## Run

```
XAI_API_KEY=... SIBYL_DB_PATH=./sibyl.db cargo run
docker build -t sibyl . && docker run -p 8090:8090 -e XAI_API_KEY=... -v sibyl-data:/data sibyl
```

## Local model

sibyl talks plain OpenAI chat completions, so any server speaking that dialect works. Verified against llama.cpp `llama-server` build b10052 hosting Qwen3.5-9B, using the liquid runtime layout:

```
cd ~/.local/share/liquid/runtime/llama-vulkan
LD_LIBRARY_PATH=. ./llama-server \
  -m ~/.local/share/liquid/runtime/models/03b74727a860-Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18099 -c 8192 --jinja
```

Then point sibyl at it and leave `XAI_API_KEY` unset:

```
SIBYL_API_BASE=http://127.0.0.1:18099/v1 SIBYL_MODEL=Qwen3.5-9B-Q4_K_M \
  SIBYL_DB_PATH=./sibyl.db cargo run
```

Notes on the llama-server side:

- `--jinja` selects the jinja template engine, which is what turns model output into OpenAI `tool_calls`. It has been the default since well before b10052, but passing it explicitly is free and keeps the command correct on older builds, where `tools` without it fails with `tools param requires --jinja flag`.
- `--host 127.0.0.1` is already the default. Keep it loopback: llama-server does no authentication unless you pass `--api-key`.
- `-c` sets the context window, `0` (the default) takes whatever the GGUF declares.
- `SIBYL_MODEL` is a label here. llama-server serves the single model it was launched with and ignores the field.
- Qwen has no hand-written tool-call parser in llama.cpp. Its chat template goes through the autoparser, which derives the tool-call format from the template itself.
- Thinking models route their thoughts into a non-standard `reasoning_content` field rather than `content`, so `content` often arrives empty next to `tool_calls`. sibyl ignores reasoning when there is real output, and falls back to showing it only when a turn came back with no content and no tool calls.

Tool calling on a 9B model is noticeably less reliable than on a frontier model: expect it to skip tools it should have called, invent argument names, and lose the thread on multi-step chains. Thinking models also sometimes mis-close their own think tags, which leaks a stray `</think>` into the answer text, seen once in a handful of local runs. sibyl passes content through as the server reports it and does not try to repair this. Keep system prompts explicit about when to call what, keep tool schemas small, and stay on the cloud model wherever a wrong tool call is expensive.

## API

- `GET /health` service liveness
- `GET /sessions` sessions, newest first
- `POST /sessions` `{"name"}`, creates and activates
- `POST /sessions/{id}/activate` makes it the active session
- `PATCH /sessions/{id}` `{"name"}` renames
- `DELETE /sessions/{id}` deletes, 400 if active
- `POST /sessions/{id}/messages` `{"content"}` appends a user message without running the model
- `POST /runs` `{"system_prompt","message"}` runs the agent loop against the active session, NDJSON stream

Run events, one JSON object per line: `text`, `tool_call`, `tool_return`, `error`, `done`. Every stream ends with `done`.

The tool manifest comes from `GET {GEOLANG_URL}/tools` and is cached for 60 seconds, tool calls go to `POST {GEOLANG_URL}/tools/{name}` with `{"args": {...}}` and read back `.result`.
