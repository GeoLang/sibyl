# sibyl

Generic LLM agent loop as a microservice: it calls an OpenAI-compatible chat completions endpoint, dispatches tool calls over HTTP to an external executor, stores session history in sqlite, and streams run events as NDJSON.

## Env

| var | default | notes |
| --- | --- | --- |
| `SIBYL_CLOUD_API_KEY` | unset | bearer for the cloud server, see below |
| `PLATFORM_JWT_SECRET` | required | shared HS256 platform secret, see below |
| `SIBYL_ALLOW_UNAUTHENTICATED` | unset | `1`/`true`/`yes`/`on` starts without the secret and leaves every session unowned |
| `SIBYL_HOST` | `0.0.0.0` | |
| `SIBYL_PORT` | `8090` | |
| `SIBYL_DB_PATH` | `/data/sibyl.db` | parent dir is created |
| `SIBYL_CLOUD_API_BASE` | `https://api.x.ai/v1` | trailing slash is stripped |
| `SIBYL_CLOUD_MODELS` | `grok-4-1-fast-reasoning` | comma separated, one profile per model |
| `SIBYL_LOCAL_API_BASE` | unset | trailing slash is stripped, unset means no local profiles |
| `SIBYL_LOCAL_MODELS` | unset | comma separated, required when `SIBYL_LOCAL_API_BASE` is set and refused when it is not |
| `SIBYL_LOCAL2_API_BASE` | unset | second local server, same rules as `SIBYL_LOCAL_API_BASE` |
| `SIBYL_LOCAL2_MODELS` | unset | comma separated, required when `SIBYL_LOCAL2_API_BASE` is set and refused when it is not |
| `GEOLANG_URL` | `http://geolang-api:8080` | tool manifest and executor |
| `SIBYL_TOOL_TIMEOUT_SECS` | `600` | per tool call |
| `SIBYL_MAX_MODEL_CALLS` | `30` | model calls per run |
| `SIBYL_RUN_BUDGET_SECS` | `900` | wall clock per run |
| `SIBYL_MAX_TOKENS` | unset | `max_tokens` per request, mainly for thinking models on small context windows, which otherwise think until the context runs out |
| `SIBYL_THINKING` | unset | `1`/`true` asks the local llama-server for thinking per request (`chat_template_kwargs`) with qwen's thinking sampling, overriding its startup `--reasoning off`. Local profiles only, cloud requests are untouched |

An empty value counts as unset and falls back to the default, so a compose `${VAR:-}` pass-through cannot blank a base URL or a model list.

### Who a session belongs to

`PLATFORM_JWT_SECRET` is the shared HS256 secret the rest of the platform validates with, over `{sub, exp}` and no `aud`. The verified `sub` owns every session that caller creates: `GET /sessions` lists only theirs, the active session is theirs alone, and naming someone else's session id on any route answers 404 rather than 403, so ids cannot be probed.

Only a plain platform bearer counts. A token carrying `token_use`, `geolang_use` or `agora_use` is refused: geolang mints the first two for its own tool boundary and its `/mcp` door, the tool ones are held by the executor that runs caller-written code, and agora mints a long lived feed token signed with the same secret.

Starting without the secret takes `SIBYL_ALLOW_UNAUTHENTICATED=1`, which logs one line at startup and leaves every session unowned and reachable by anyone who can reach the port. That is the standalone stack and the eval harness, neither of which holds a token.

Sessions written before sibyl had owners carry no subject, so no authenticated caller can reach them. They are still there for an unauthenticated start.

### Model profiles

sibyl builds one profile per model on any number of named providers and you switch between them at runtime. Env seeds at most three:

- **cloud**, one profile per entry of `SIBYL_CLOUD_MODELS`, calling `SIBYL_CLOUD_API_BASE` with `SIBYL_CLOUD_API_KEY`.
- **local**, one profile per entry of `SIBYL_LOCAL_MODELS`, calling `SIBYL_LOCAL_API_BASE` with no key.
- **local2**, the same for `SIBYL_LOCAL2_MODELS` and `SIBYL_LOCAL2_API_BASE`, for a second local server.

Settings can add more of either kind (`PUT /model/providers`): another cloud API with its own base and key, or another local llama-server with its own base and model list. `DELETE /model/providers/{id}` removes one. Both routes need a platform bearer when the gate is on, the same as `PUT /model/cloud`. The list is stored in sqlite and overrides env on the next start.

A profile id is `<provider>:<model>`, so `local:Qwen3.5-9B-Q4_K_M`, `local2:gpt-oss-20b` and `cloud:grok-4-1-fast-reasoning`, and its label is `<model> (<provider>)`. `GET /models` lists the local profiles first, then the cloud ones, and includes `cloud.base`, `cloud.models` and `cloud.has_key` so the viewer can show the current cloud server without echoing the key. Local profiles also carry `reachable`, from a short `GET {base}/models` probe, so a turned-off host is greyed out rather than looking live. A run on a local profile whose host is down fails with a message and keeps that profile active: sibyl never switches to a cloud API on its own.

Without `SIBYL_CLOUD_API_KEY` the cloud profiles are still listed and marked unavailable, so a viewer can grey them out rather than hide them, and switching to one answers 409. The process still starts: a run then fails until Settings saves a key or a local server is configured. `SIBYL_LOCAL_MODELS` and `SIBYL_LOCAL_API_BASE` have to be set together: either one alone fails startup naming both, and the `SIBYL_LOCAL2_` pair is checked the same way.

The first local profile is active by default, the first cloud one when there is no local server. That choice is stored in sqlite and survives a restart. A stored id naming a profile that is now unavailable or no longer configured, say the key was pulled or the model left the list, falls back to the default rather than starting broken.

`PUT /model` switches the active profile. `PUT /model/cloud` rewrites the cloud base, key and model list (`{"base","key","models"}`, each optional) and switches to the first new cloud profile. A missing field keeps the current value; an empty key is 400. The new values are stored in sqlite and override env on the next start. The route needs a platform bearer when the gate is on, because the published port would otherwise take a key from anyone who can reach it.

The key only ever goes to `SIBYL_CLOUD_API_BASE`. Local profiles send no `Authorization` header and log one line per local server at startup saying so, so point `SIBYL_LOCAL_API_BASE` and `SIBYL_LOCAL2_API_BASE` at servers you trust on a network you trust. The viewer never sees the key: `GET /models` has `has_key` and no `key` field.

## Run

```
SIBYL_CLOUD_API_KEY=... SIBYL_DB_PATH=./sibyl.db cargo run
docker build -t sibyl . && docker run -p 8090:8090 -e SIBYL_CLOUD_API_KEY=... -v sibyl-data:/data sibyl
```

## Local model

sibyl talks plain OpenAI chat completions, so any server speaking that dialect works. Verified against llama.cpp `llama-server` build b10052 hosting Qwen3.5-9B, using the liquid runtime layout:

```
cd ~/.local/share/liquid/runtime/llama-vulkan
LD_LIBRARY_PATH=. ./llama-server \
  -m ~/.local/share/liquid/runtime/models/03b74727a860-Qwen3.5-9B-Q4_K_M.gguf \
  -a Qwen3.5-9B-Q4_K_M \
  --host 127.0.0.1 --port 18099 -c 8192 --jinja
```

Then point sibyl at it and leave `SIBYL_CLOUD_API_KEY` unset:

```
SIBYL_LOCAL_API_BASE=http://127.0.0.1:18099/v1 SIBYL_LOCAL_MODELS=Qwen3.5-9B-Q4_K_M \
  SIBYL_DB_PATH=./sibyl.db cargo run
```

Notes on the llama-server side:

- `--jinja` selects the jinja template engine, which is what turns model output into OpenAI `tool_calls`. It has been the default since well before b10052, but passing it explicitly is free and keeps the command correct on older builds, where `tools` without it fails with `tools param requires --jinja flag`.
- `--host 127.0.0.1` is already the default. Keep it loopback: llama-server does no authentication unless you pass `--api-key`.
- `-c` sets the context window, `0` (the default) takes whatever the GGUF declares.
- `-a` (`--alias`) sets the name `GET /v1/models` reports, which is what makes the `SIBYL_LOCAL_MODELS` entry and the served name agree. A single model server ignores the `model` field of a request and answers with the one it loaded either way, so the alias is for the listing, not for routing.
- Qwen has no hand-written tool-call parser in llama.cpp. Its chat template goes through the autoparser, which derives the tool-call format from the template itself.
- Thinking models route their thoughts into a non-standard `reasoning_content` field rather than `content`, so `content` often arrives empty next to `tool_calls`. sibyl ignores reasoning when there is real output, and falls back to showing it only when a turn came back with no content and no tool calls.

### Several local models from one server

`--models-dir` starts llama-server in router mode, where it lists every GGUF in the directory and picks one per request by the `model` field:

```
cd ~/.local/share/liquid/runtime/llama-vulkan
LD_LIBRARY_PATH=. ./llama-server \
  --models-dir ~/.local/share/liquid/runtime/models \
  --host 127.0.0.1 --port 18099
```

The router names each model after its GGUF filename with the `.gguf` dropped, and those names are what `SIBYL_LOCAL_MODELS` lists:

```
SIBYL_LOCAL_API_BASE=http://127.0.0.1:18099/v1 \
  SIBYL_LOCAL_MODELS=Qwen3.5-9B-Q4_K_M,Qwen3.8-27B-Q4_K_M \
  SIBYL_DB_PATH=./sibyl.db cargo run
```

Qwen 3.8 ships a 27B and no 9B, so the small model stays Qwen 3.5: `Qwen3.8-27B-Q4_K_M.gguf` is the bartowski Q4_K_M file (about 18 GB), and the 9B is the `Qwen3.5-9B-Q4_K_M.gguf` the liquid runtime already holds.

The liquid runtime prefixes its downloads with a content hash, which is part of the filename and so part of the name: `03b74727a860-Qwen3.5-9B-Q4_K_M`. `GET /v1/models` on the router reports the exact ids.

- Loading is on demand and `--models-max` caps how many stay resident, 4 by default. `--no-models-autoload` turns the on demand load off, and a request then only reaches a model already loaded.
- A `model` the router does not know is a 400 `model 'x' not found`, so a typo in `SIBYL_LOCAL_MODELS` fails the run rather than answering from the wrong model.
- The router builds each child llama-server command itself, passing only the model path and its alias. Per model flags, a context size or `--n-cpu-moe`, go in the INI file `--models-preset` names.

Tool calling on a 9B model is noticeably less reliable than on a frontier model: expect it to skip tools it should have called, invent argument names, and lose the thread on multi-step chains. Thinking models also sometimes mis-close their own think tags, which leaks a stray `</think>` into the answer text, seen once in a handful of local runs. sibyl passes content through as the server reports it and does not try to repair that.

The one exception is tool-call salvage. A small model sometimes prints a tool call as literal text instead of emitting a structured one, and llama-server's parser hands the raw markup straight through to the screen. When a turn comes back with no structured tool calls and its text contains a complete `<tool_call>` block, sibyl parses it and runs it as a normal call, logging a warn naming the dialect. Two dialects are recognized, the Qwen-Coder XML form (`<function=name><parameter=key>value</parameter></function>`) and the Hermes JSON form (`{"name": ..., "arguments": {...}}`). Parsing is strict and never repairs: a block that is incomplete, ambiguous, or names a tool outside the current manifest is left in the text exactly as it arrived. A turn that already carries structured tool calls is never touched, so the cloud path is unaffected. Watch for those warns, since a run that depends on salvage is a sign the model is drifting off its template.

Keep system prompts explicit about when to call what, keep tool schemas small, and stay on a cloud profile wherever a wrong tool call is expensive.

### A second local server

`SIBYL_LOCAL2_API_BASE` and `SIBYL_LOCAL2_MODELS` seed a second local provider whose id and label are `local2`, so its profiles read `local2:<model>`. The pair follows the same rules as the first: both set or neither, trailing slash stripped, no `Authorization` header, and the same `GET {base}/models` reachability probe. Use it when two llama-servers run at once, say a small model on this box and a larger one over the tunnel below.

Both env pairs are seeds only. Once a provider list is stored in sqlite, by the Settings panel or `PUT /model/providers`, that list wins and the env pairs are ignored, and startup logs say so. On a deployment with stored providers, add a server with `PUT /model/providers` instead of env, it takes effect with no restart.

```
SIBYL_LOCAL_API_BASE=http://127.0.0.1:18099/v1 SIBYL_LOCAL_MODELS=Qwen3.5-9B-Q4_K_M \
  SIBYL_LOCAL2_API_BASE=http://host.docker.internal:18200/v1 SIBYL_LOCAL2_MODELS=Qwen3.5-35B-A3B \
  SIBYL_DB_PATH=./sibyl.db cargo run
```

### Model on another machine (shared with liquid)

To serve the model from a remote box that also runs [liquid](https://github.com/boxerab/liquid), share liquid's own managed server instead of loading a second model. Two pieces:

1. On the remote machine, pin liquid's inference port in `~/.config/liquid/liquid.toml` so it stops picking a random one:

   ```toml
   [inference]
   tier = "gpu-mid"
   port = 18200
   ```

   liquid launches llama-server on `127.0.0.1:18200` when its inference is in use. b10052+ has `--jinja` on by default, so OpenAI tool calls work, and liquid's `--reasoning off` is harmless here. The model is whatever liquid's tier says, and a single model server ignores the `model` field, so the `SIBYL_LOCAL_MODELS` entry is only the name the profile shows. liquid runs `--parallel 1`, so sibyl runs and liquid's own jobs queue behind each other, and the local profiles only work while liquid's server is up. Switch to a cloud profile in the viewer settings when it isn't.

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

   `systemctl --user enable --now geolang-llama-tunnel`, then start sibyl with `SIBYL_LOCAL_API_BASE=http://host.docker.internal:18200/v1` and a `SIBYL_LOCAL_MODELS` entry naming the tier's model.

For a dedicated always-on server on the remote box instead (a second loaded model, but independent of liquid's lifecycle), use the launch line from the section above over SSH with `nohup`, and keep `--n-cpu-moe` for MoE tiers per liquid's own tuning.

## API

- `GET /health` service liveness
- `GET /sessions` the caller's sessions, newest first
- `POST /sessions` `{"name"}`, creates and activates
- `POST /sessions/{id}/activate` makes it the caller's active session
- `PATCH /sessions/{id}` `{"name"}` renames
- `DELETE /sessions/{id}` deletes, 400 if active
- `POST /sessions/{id}/messages` `{"content"}` appends a user message without running the model
- `GET /models` `{"active", "profiles":[{"id","label","model","server","provider","available","reachable"}], "providers":[{"id","label","server","base","models","has_key","reachable"}], "cloud":{"id","base","models","has_key"}}`, local profiles first
- `PUT /model` `{"id"}` switches profile, 204 on success, 404 for an unknown id, 409 when that profile is unavailable
- `PUT /model/cloud` `{"base"?,"key"?,"models"?}` rewrites the cloud provider and switches to its first profile, 204, 400 for an empty key
- `PUT /model/providers` `{"id"?,"label"?,"server"?,"base"?,"key"?,"models"?}` adds or updates one provider, 204
- `DELETE /model/providers/{id}` removes one provider, 204
- `POST /runs` `{"system_prompt","message","user_token"?,"thread_id"?,"document"?,"profile"?,"without_tools"?}` runs the agent loop, NDJSON stream. `thread_id` is the session id (AG-UI thread); when it is absent the caller's own active session is used, and one is created when they have none. A `thread_id` naming someone else's session ends the stream with an `error` event. `user_token` is the caller's bearer token: it names the session owner, and it is sent as `Authorization: Bearer` on every tool call of that run and kept in memory only. `document` is the agora document the asker is looking at, sent as `X-Agora-Document` on every tool call of the run so a tool can read that map. `profile` pins this one run to an exact profile id, `local:Qwen3.5-35B-A3B` and the like, as `GET /models` lists them: that profile's client answers the run, and the active profile is neither read nor changed. An id that is unknown or has no client is a 400 naming it. Absent, the run uses the active profile as before. `without_tools` names manifest tools the model is not offered on this run, which is how geolang keeps a model from reaching for an older tool its viewer catalogue replaces. The two memory tools below are added after the filter, so they cannot be left out.

Every `/sessions` route reads the bearer from the `Authorization` header, and `/runs` reads the same token from `user_token`. With `PLATFORM_JWT_SECRET` set, a missing or invalid one is a 401 with `{"error": ...}`. Without the secret nothing is checked and the tools call services unauthenticated.

`/models`, `/model` and `/health` carry no gate: put sibyl behind something that authenticates if the model switch matters.

A switch applies to the next run. A run already going finishes on the profile it started with.

Run events, one JSON object per line: `text`, `tool_call`, `tool_return`, `error`, `done`. Every stream ends with `done`. A run that hits `SIBYL_MAX_MODEL_CALLS` or `SIBYL_RUN_BUDGET_SECS` ends with an `error` event naming which one it was. So does a tool called three times in a row with the same arguments and the same result, whether it succeeded or failed. When a run ends on an error, sibyl appends an assistant message `I could not finish: <error>` to the session, so the next prompt reads the earlier request as closed rather than finishing it unasked.

Dropping the NDJSON connection cancels the run. sibyl requests completions with `stream: true` and accumulates them server side, so a dropped run also drops the in-flight request and the model server stops generating instead of finishing an answer nobody is waiting for. The loop checks for a departed client before each model call too, in case the disconnect lands between calls. The event shape is unchanged either way, sibyl does not forward partial tokens.

The tool manifest comes from `GET {GEOLANG_URL}/tools` and is cached for 60 seconds, tool calls go to `POST {GEOLANG_URL}/tools/{name}` with `{"args": {...}}` and read back `.result`.

## Session memory

Beside the manifest, sibyl offers two tools of its own and runs them itself rather than sending them to the executor. `save_memory` stores one short fact against the session, `forget_memory` deletes the stored facts containing a given string. Up to 50 of a session's facts are appended to the system prompt on every later turn of that session, so recall costs no tool call. Facts do not cross sessions.

The rest of the context is trimmed for you. Once a request is estimated over 100000 tokens, the last 20 messages stay verbatim and everything older goes to the run's own model to be summarized into one line of the system prompt, which is stored on the session so it is not summarized again. A summarization that fails drops the oldest messages for that request only. A tool result over 20000 characters is cut there and ends with `[truncated]`, in the `tool_return` event as well as in the history.
