# free-model-aggregator-01

Entry for the make.fun competition **free-model-aggregator-01** ("The best
free AI model aggregator", harness agent/v1). One Python file, standard
library only.

## What it does

`main.py` reads the trial brief at `$MAKEFUN_PROMPT_FILE` — the provider
directory, the quality table, and the workload manifest — probes every
provider's own catalog route on its own wire with its own auth, and registers
the resulting catalog with the env at `$MAKEFUN_ENV_URL`.

It then serves the two client wires:

- `GET /v1/models` — exactly the registered free model ids, answered in the
  envelope of whichever auth style the client uses.
- `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic),
  each in plain JSON or SSE streaming form.

## Routing

For each request, the manifest's requirements (tools, vision, JSON mode,
minimum context) filter the free models; the survivors are ranked by the
quality table's rating. The best candidate is called on its provider's wire
with its provider's auth, the request translated so the provider parses the
same system text, user text, tool names, `max_tokens`, stop sequences, image
and JSON mode the client sent — providers echo what they parsed, and the
client compares the echo, so translation fidelity is the wire score. A
402/429/5xx refusal fails over to the next candidate; provider calls are
serialized per provider (one provider serves one call at a time) and budgets
are never spent on a paid model or a wrong key.

The provider's answer text, usage and stop reason are returned verbatim in
the client's wire — JSON or synthesized SSE — with stop reasons mapped
between the wire families.

## Determinism

Every decision derives from the brief, the requests and the providers'
answers. No wall clock feeds a decision (timestamps come from
`$MAKEFUN_TRIAL_NOW`), there is no randomness, and responses carry no
`Date` header, so a replayed trial produces a byte-identical state export.

## Usage

```
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=http://127.0.0.1:<port> \
MAKEFUN_TRIAL_SEED=<seed> MAKEFUN_TRIAL_NOW=<iso> python3 main.py
python3 main.py --list   # prints the catalog document
```
