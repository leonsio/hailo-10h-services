# Frigate assistant

Frigate-Assist intent recognition, deterministic time/camera resolution, tool/history planning, prompt compaction and result validation. `frigate_context.py` centralizes shared timestamps, camera matching and tool context. `frigate_shortcuts.py` finishes high-confidence read-only rounds such as exact recap listings and named-person sighting evidence without invoking an LLM. Requests retain their camera catalogue and output contracts. Unmatched or interpretive forms continue to the model. Grammar, synonyms and responses live in `../../locales/`.
