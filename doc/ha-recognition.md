# HA recognition, diagnosis and caching

The goal is fewer wrong actions and smaller correct model inputs. Exact intents
and direct state/value paths remain first-class. HA-Assist calls zero generative
backends when it can solve a request, otherwise exactly one configured text or
image backend. All HA behaviour stays at the virtual-model boundary.

## Recognition policy

1. Canonical catalogue spellings/configured aliases, explicit domain and area
   constrain eligible targets. Current request data, not an installation-specific
   name list, is authoritative.
2. Character similarity and Damerau-Levenshtein adjacent-transposition evidence
   rank remaining catalogue spans. Similarity is not a calibrated probability.
   Exact spans are never consumed by overlapping fuzzy replacements.
3. HassIL validates the complete sentence and actual client tool schema. On a
   miss, recovery can rank official sentence skeletons with existing explicit
   targets/values bound into slots. No entity-name permutations are generated.
4. Sentence recovery requires explicit action evidence. Negation, conditions,
   relative changes, multiple values and combined actions do not enter recovery.
   Opposite on/off actions are never interchangeable. Known article forms come
   from the official grammar rather than manually listing input variants.
5. Long target names are excluded from the structure score so that a short changed
   qualifier cannot be hidden by a high whole-sentence score. All accepted
   reconstructions are reparsed. Distinct validated calls still need a score
   margin; excessive candidate sets fall back rather than dropping competitors.
6. Direct and generated calls both obey semantic target/value/capability checks.
   An explicit named device cannot become a room-wide action.

Configuration (also available as `HAILO_...` environment overrides):

```yaml
settings:
  ha_assist_fuzzy_enabled: true
  ha_assist_fuzzy_threshold: 90.0
  ha_assist_fuzzy_margin: 8.0
  ha_assist_sentence_fuzzy_enabled: true
  ha_assist_sentence_threshold: 94.0
```

Sentence threshold accepts 90–100; 94 is conservative by default. Recovery stores
at most 32 language/tool-set indexes and up to 4096 skeletons per index with a
per-intent quota. Exact/direct success does not build the sentence index. Aliases
are configured HA aliases, never automatically learned typo variants. Optional
structured metadata is documented in [the API](api.md).

## Diagnosis and regression evaluation

`POST /v1/ha-assist/diagnose` accepts a normal HA-Assist request. It returns
candidate/action evidence, proposed responses, preparation timings and the entire
prepared message/tool payload. No generative inference or tool execution occurs;
MiniLM may still be used for retrieval on its existing serialized owner thread.
The response explicitly labels its prompt stage before native rendering/budgeting.
It uses existing API authentication and request limits.

Tests cover synthetic changing catalogues, typo deletions/transpositions/spaces,
ambiguous candidates, configured aliases, named-target preservation, capability
rejection, protected qualifiers and absence of generative calls during diagnosis.
Hypothesis generates unseen catalogue strings instead of maintaining typo lists.
Track false actions, intent/target accuracy, correct clarification, generative
fallback rate and P50/P95 latency separately; a lower fallback rate alone is not
proof of improvement. Real backend timings still require a Hailo/LiteRT host.

## What tool caching changes

The static tool index caches immutable lexical schema fields and corpus-frequency
features. The key includes complete canonical schema JSON and language; changed
schemas/enums invalidate naturally. It holds at most 64 entries and bypasses
caching for keys larger than 64 KiB. Selected tools and pruned enums remain fresh
copies. Existing MiniLM embedding caches are bounded at 512 entries.

This avoids repeated parsing/tokenization/embedding work. It does **not** cache
LLM answers, executable actions or current states, and does not remove schema
tokens from the native prompt. MiniLM ranks a small relevant tool/entity set;
that selection can reduce LLM prefill more materially than a warm Python cache.
It is retrieval evidence, never permission to change a number, target or polarity.

## Native backend findings (checked 2026-10-06)

- [HailoRT 5.4.0 Python bindings](https://github.com/hailo-ai/hailort/blob/v5.4.0/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py)
  expose `save_context()` and `load_context()` for both LLM and VLM. The opaque
  binary snapshot includes conversation history and is tied to its originating
  model. This permits context switching; it is not an automatic arbitrary-prefix
  cache. The service clears context before/after each native request and keeps
  snapshot reuse disabled. No extra generation is used to warm a prefix.
- Current LiteRT-LM sources include
  [PrefixCache](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/core/prefix_cache.h)
  and [CachedSession](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/core/cached_session.cc),
  which find shared token/media prefixes, rewind and prefill the remaining suffix.
  The inspected [Python Engine API](https://github.com/google-ai-edge/LiteRT-LM/blob/main/python/litert_lm/engine.py)
  exposes conversations and an artifact `cache_dir`; that directory must not be
  presented as a user-conversation KV cache switch. Presence in current sources
  does not prove cross-conversation reuse in an installed wheel/model. The service
  creates isolated conversations, leaves internal runtime behaviour to the SDK
  and does not advertise measured prefix-cache savings.

`/health.cache` inspects currently loaded method availability without saving or
loading context. A real A/B benchmark must distinguish cold/warm prefill and
cached-token counts before enabling service-managed reuse. Cache identity would
need model/template/schema/prefix identity and conversation isolation; stale HA
state must not be carried over. Smaller task-specific prompts remain the default,
without forwarding extra catalogue data merely to produce a longer common prefix.

## Does an HA API adapter help?

A deterministic read-only state mirror could supply exposed states via HA
WebSocket events. It would be a data adapter, not another generative agent. Its
largest benefit would be avoiding the LLM's initial decision to fetch live data
and subsequent result-formatting inference for queries it can answer directly.
Our existing direct GetLiveContext path already avoids those model calls for
supported questions; there a mirror mainly saves one fast HA round trip.

A mirror also needs exposure/permission filtering, timestamps, unknown/unavailable
handling, reconnection resynchronization and invalidation after actions. It must
never guess freshness. Therefore no automatic HA connection or state-answer cache
is introduced here. Optional structured catalogue metadata enables a future
adapter without requiring HA credentials or changing legacy clients. If real
measurements show remaining state questions spending time in a generative round
trip, add the event-fed read-only adapter then. Home Assistant continues executing
all returned actions.

## Related design references

- [HassIL](https://github.com/OHF-Voice/hassil) and
  [official HA intents](https://github.com/OHF-Voice/intents): grammar, slots and
  final intent validation; these are existing dependencies.
- [hass-closest-intent](https://github.com/charludo/hass-closest-intent): canonical
  sentence reconstruction and diagnostics as design references.
- [Assist Canonicalizer](https://github.com/luuquangvu/assist-canonicalizer):
  independent ranking signals, bounded candidate indexes and explicit confidence
  gates as design references. This implementation does not vendor those projects.
