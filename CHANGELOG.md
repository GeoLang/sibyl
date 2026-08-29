# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- 2026-08-28: **any number of cloud APIs and local servers.** Each is a named
  provider with its own base, optional key and model list. Env still seeds the
  `cloud` and `local` providers. `PUT /model/providers` adds or updates one,
  `DELETE /model/providers/{id}` removes one, and `PUT /model/cloud` still
  rewrites the `cloud` provider. `GET /models` includes a `providers` array
  (base and `has_key`, never the key).
- 2026-08-28: **cloud credentials can be rewritten at runtime.** `PUT /model/cloud`
  takes optional `base`, `key` and `models`, stores them in sqlite (overriding
  env on the next start), rebuilds the cloud profiles and switches to the first
  new one. The route needs a platform bearer when the gate is on. `GET /models`
  now includes `cloud.base`, `cloud.models`, `cloud.has_key` and
  `local_reachable`, never the key, and marks a local profile unreachable when
  a short `GET {base}/models` probe fails, so a turned-off host is a 409 rather
  than a hung chat. `SIBYL_CLOUD_API_KEY` is no longer required to start: with
  neither a key nor a local server the cloud profiles are listed unavailable
  and a run tells you to pick a model in Settings.

### Changed

- 2026-08-29: **a call repeated with the same result stops the run.** The
  repeat guard used to count only identical failures. A tool called three
  times in a row with the same arguments and the same result now aborts the
  run too, so a model that keeps re-issuing a call that already did its job
  cannot burn the budget.
- 2026-08-29: **a failed run leaves a closing note in the session.** When the
  model call or a tool fails partway, sibyl appends an assistant message
  "I could not finish: <error>" before ending the stream, so the next prompt
  reads the earlier request as closed instead of finishing it unasked.
- 2026-08-29: **a down local host fails the run instead of switching to cloud.**
  `client_for_run` used to activate the first cloud profile and persist it when
  the active local host did not answer, so a prompt meant for a local model
  went to the cloud API. Now the run fails with the same message the chat shows
  and the active profile stays put.
- 2026-08-26: **one switchable profile per model, on a cloud server and a local
  one**. `XAI_API_KEY`, `SIBYL_API_BASE` and `SIBYL_MODEL` are gone, replaced by
  `SIBYL_CLOUD_API_KEY`, `SIBYL_CLOUD_API_BASE` (default `https://api.x.ai/v1`),
  `SIBYL_CLOUD_MODELS` (default `grok-4-1-fast-reasoning`),
  `SIBYL_LOCAL_API_BASE` and `SIBYL_LOCAL_MODELS`. The two model lists are comma
  separated and give one profile per entry, so the cloud server is no longer
  pinned to x.ai and a local llama-server in router mode can serve several
  models at once. A profile id is now `<server>:<model>` and its label is
  `<model> (<server>)`, and `GET /models` carries a new `server` field of
  `cloud` or `local`, listing local profiles first. A stored `active_model` of
  `cloud` or `local` from an older build no longer names anything and falls back
  to the default profile, the first local one or else the first cloud one.
  Cloud profiles are listed even with no key, marked unavailable, so a viewer
  can grey them out. Setting `SIBYL_LOCAL_API_BASE` without `SIBYL_LOCAL_MODELS`
  or the other way round fails startup naming both, and so does having neither
  a key nor a local base. `SIBYL_THINKING` still reaches local profiles only and
  the cloud key still never goes to the local server.

- 2026-08-25: **an agora feed token is refused as a session bearer**. `subject`
  already rejected `token_use` and `geolang_use`, and now rejects `agora_use`
  with any value. agora mints a long lived feed token signed with the same
  `PLATFORM_JWT_SECRET` and carrying no `role`, so before this it opened
  sessions owned by the feed's uuid.

- 2026-08-25: **jsonwebtoken 9 to 11 on the `aws_lc_rs` backend**, the same
  crypto crate reqwest's rustls already pulls in. Only HS256 `encode`,
  `decode`, `from_secret` and `Validation::default()` are used, and 11 keeps
  `validate_aud` on by default, so a token carrying `aud` is still refused.

### Added

- 2026-08-25: **a run can name the map the asker is looking at**. `POST /runs`
  takes an optional `document`, and every tool call of that run carries it as
  `X-Agora-Document` beside the bearer, so a tool that reads a live map answers
  about the one on screen. Absent or empty sends no header.

- 2026-08-25: **sessions have an owner**. sibyl reads `PLATFORM_JWT_SECRET`, the
  same HS256 platform secret every other service validates with, and refuses to
  start without it unless `SIBYL_ALLOW_UNAUTHENTICATED` is set to `1`, `true`,
  `yes` or `on`, which logs one line saying sessions are unowned. The verified
  `sub` of the bearer owns every session that caller creates: it is stored in a
  new nullable `subject` column, added by an idempotent `ALTER TABLE` on open.
  `GET /sessions` lists only the caller's own, and the active session is per
  subject, so creating or activating one no longer switches the session someone
  else is reading. `activate`, `rename`, `delete` and `POST
  /sessions/{id}/messages` answer 404 for a session belonging to another
  subject, the same answer a missing id gets, so ids cannot be probed. `POST
  /runs` takes the subject from `user_token`: a run with no `thread_id` uses the
  caller's own active session, and a `thread_id` naming someone else's session
  fails as not found. Every `/sessions` route and `/runs` answer 401 with
  `{"error": ...}` when the token is missing or invalid. Sessions written before
  the column keep a NULL subject and are reachable only with the gate off.
  A token carrying `token_use` or `geolang_use` is refused: those are geolang's
  scoped tool credential and its `/mcp` token, and the executor that runs
  caller-written code holds the tool ones.
