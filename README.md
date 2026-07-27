# sibyl

Generic LLM agent loop as a microservice: it calls an OpenAI-compatible chat completions endpoint, dispatches tool calls over HTTP to an external executor, stores session history in sqlite, and streams run events as NDJSON.

## Env

| var | default | notes |
| --- | --- | --- |
| `XAI_API_KEY` | required | startup fails without it |
| `SIBYL_HOST` | `0.0.0.0` | |
| `SIBYL_PORT` | `8090` | |
| `SIBYL_DB_PATH` | `/data/sibyl.db` | parent dir is created |
| `SIBYL_MODEL` | `grok-4-1-fast-reasoning` | |
| `SIBYL_API_BASE` | `https://api.x.ai/v1` | |
| `GEOLANG_URL` | `http://geolang-api:8080` | tool manifest and executor |
| `SIBYL_TOOL_TIMEOUT_SECS` | `600` | per tool call |

## Run

```
XAI_API_KEY=... SIBYL_DB_PATH=./sibyl.db cargo run
docker build -t sibyl . && docker run -p 8090:8090 -e XAI_API_KEY=... -v sibyl-data:/data sibyl
```

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
