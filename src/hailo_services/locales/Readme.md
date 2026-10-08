# Language catalogues

One complete JSON file per language: de, en, ru, fr, es, it, nl and pt. Edit replies in `text`, browser labels in `ui`, HA vocabulary in `aliases`, HA supplemental grammar in `ha_intents`, and Frigate grammar/vocabulary/hints in `frigate`. Keep placeholder names aligned. `de.json.routing` contains shared canonical matching rules. Supported languages are discovered from filenames. Restart the service after edits; no Python changes are needed for existing text/grammar edits.
