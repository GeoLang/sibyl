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
| `SIBYL_MAX_MODEL_CALLS` | `30` | model calls per run |
| `SIBYL_RUN_BUDGET_SECS` | `900` | wall clock per run |
| `SIBYL_MAX_TOKENS` | unset | `max_tokens` per request, mainly for thinking models on small context windows, which otherwise think until the context runs out |
| `SIBYL_THINKING` | unset | `1`/`true` asks the local llama-server for thinking per request (`chat_template_kwargs`) with qwen's thinking sampling, overriding its startup `--reasoning off`. Local profile only, cloud requests are untouched |

An empty value counts as unset and falls back to the default, so a compose `${VAR:-}` pass-through cannot blank the base URL or the model.

### Model profiles

sibyl builds two profiles at startup and you switch between them at runtime.

- **cloud**, the default x.ai endpoint, available only when `XAI_API_KEY` is set.
- **local**, `SIBYL_API_BASE` and `SIBYL_MODEL`, available only when `SIBYL_API_BASE` names something other than the x.ai URL.

Set both and you get both, with local active by default. Set only one and sibyl behaves as it always has, including failing at startup when neither works out, which keeps a forgotten key loud. `SIBYL_MODEL` names the local model when a local base is configured, and the cloud model otherwise.

The active profile is stored in sqlite and survives a restart. A stored choice that is no longer available, say the key was pulled, falls back to whatever is available rather than starting broken.

A local profile with no key sends no `Authorization` header and logs one line saying so. A key set alongside a custom base still goes to that host, so only point sibyl at servers you trust.

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

Tool calling on a 9B model is noticeably less reliable than on a frontier model: expect it to skip tools it should have called, invent argument names, and lose the thread on multi-step chains. Thinking models also sometimes mis-close their own think tags, which leaks a stray `</think>` into the answer text, seen once in a handful of local runs. sibyl passes content through as the server reports it and does not try to repair that.

The one exception is tool-call salvage. A small model sometimes prints a tool call as literal text instead of emitting a structured one, and llama-server's parser hands the raw markup straight through to the screen. When a turn comes back with no structured tool calls and its text contains a complete `<tool_call>` block, sibyl parses it and runs it as a normal call, logging a warn naming the dialect. Two dialects are recognized, the Qwen-Coder XML form (`<function=name><parameter=key>value</parameter></function>`) and the Hermes JSON form (`{"name": ..., "arguments": {...}}`). Parsing is strict and never repairs: a block that is incomplete, ambiguous, or names a tool outside the current manifest is left in the text exactly as it arrived. A turn that already carries structured tool calls is never touched, so the cloud path is unaffected. Watch for those warns, since a run that depends on salvage is a sign the model is drifting off its template.

Keep system prompts explicit about when to call what, keep tool schemas small, and stay on the cloud model wherever a wrong tool call is expensive.

### Model on another machine (shared with liquid)

To serve the model from a remote box that also runs [liquid](https://github.com/boxerab/liquid), share liquid's own managed server instead of loading a second model. Two pieces:

1. On the remote machine, pin liquid's inference port in `~/.config/liquid/liquid.toml` so it stops picking a random one:

   ```toml
   [inference]
   tier = "gpu-mid"
   port = 18200
   ```

   liquid launches llama-server on `127.0.0.1:18200` when its inference is in use. b10052+ has `--jinja` on by default, so OpenAI tool calls work; liquid's `--reasoning off` is harmless here. The model is whatever liquid's tier says, `SIBYL_MODEL` is only a label. liquid runs `--parallel 1`, so sibyl runs and liquid's own jobs queue behind each other, and the local profile only works while liquid's server is up. Switch to the cloud profile in the viewer settings when it isn't.

2. On the machine running the stack, a systemd user unit holds an SSH tunnel from the docker bridge to the remote loopback, so nothing is exposed on the LAN and the container reaches it as `host.docker.internal`:

   ```ini
   # ~/.config/systemd/user/geolang-llama-tunnel.service
   [Unit]
   Description=SSH tunnel to the remote llama-server (sibyl local model)
   After=network-online.target

   [Service]
   ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o BatchMode=yes -L 172.17.0.1:18200:127.0.0.1:18200 aaron@hercules
   Restart=always
   RestartSec=3

   [Install]
   WantedBy=default.target
   ```

   `systemctl --user enable --now geolang-llama-tunnel`, then start sibyl with `SIBYL_API_BASE=http://host.docker.internal:18200/v1` and a `SIBYL_MODEL` label matching the tier.

For a dedicated always-on server on the remote box instead (a second loaded model, but independent of liquid's lifecycle), use the launch line from the section above over SSH with `nohup`, and keep `--n-cpu-moe` for MoE tiers per liquid's own tuning.

## API

- `GET /health` service liveness
- `GET /sessions` sessions, newest first
- `POST /sessions` `{"name"}`, creates and activates
- `POST /sessions/{id}/activate` makes it the active session
- `PATCH /sessions/{id}` `{"name"}` renames
- `DELETE /sessions/{id}` deletes, 400 if active
- `POST /sessions/{id}/messages` `{"content"}` appends a user message without running the model
- `GET /models` `{"active", "profiles":[{"id","label","model","available"}]}`
- `PUT /model` `{"id"}` switches profile, 204 on success, 404 for an unknown id, 409 when that profile is unavailable
- `POST /runs` `{"system_prompt","message","user_token"?}` runs the agent loop against the active session, NDJSON stream. `user_token` is the caller's bearer token, sent as `Authorization: Bearer` on every tool call of that run and kept in memory only. Without it the tools call services unauthenticated.

These endpoints are unauthenticated, like the rest of sibyl's API, which already exposes run execution: put it behind something that authenticates.

A switch applies to the next run. A run already going finishes on the profile it started with.

Run events, one JSON object per line: `text`, `tool_call`, `tool_return`, `error`, `done`. Every stream ends with `done`. A run that hits `SIBYL_MAX_MODEL_CALLS` or `SIBYL_RUN_BUDGET_SECS` ends with an `error` event naming which one it was.

Dropping the NDJSON connection cancels the run. sibyl requests completions with `stream: true` and accumulates them server side, so a dropped run also drops the in-flight request and the model server stops generating instead of finishing an answer nobody is waiting for. The loop checks for a departed client before each model call too, in case the disconnect lands between calls. The event shape is unchanged either way, sibyl does not forward partial tokens.

The tool manifest comes from `GET {GEOLANG_URL}/tools` and is cached for 60 seconds, tool calls go to `POST {GEOLANG_URL}/tools/{name}` with `{"args": {...}}` and read back `.result`.
