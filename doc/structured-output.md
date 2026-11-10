# Home Assistant: Generate data

The general `/v1/chat/completions` API supports free text and structured JSON.
A native decision model is not required for `ai_task.generate_data`.

## Setup

1. In **Local OpenAI LLM**, use server type **Generic**, your service URL ending
   in `/v1`, and the API key configured in the service (if enabled).
2. Add an **AI Task** entity with text/data generation enabled. Select the direct
   model **`gemma-4-E2B-it`** and enable Gemma in the service configuration.
3. In Home Assistant's **Generate data** action, select that entity and enter
   instructions. Define a data structure for typed fields, or omit it for text.
4. Set the response token limit to at most **1024**. Input limits must leave room
   for the instructions, schema, attachments and output.

`HA-Assist` and `Frigate-Assist` are conversation/action routers. They reject JSON
response formats; use a direct chat model for structured generation instead.

## API example

```json
{
  "model": "gemma-4-E2B-it",
  "messages": [{"role": "user", "content": "Extract the room: kitchen"}],
  "max_tokens": 256,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "RoomData",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"room": {"type": "string"}},
        "required": ["room"],
        "additionalProperties": false
      }
    }
  }
}
```

Supported formats are `text`, `json_object`, and `json_schema`. A schema has a
required name and schema, plus optional description and strict flag. Both strict
and non-strict schema requests validate the final output against the supplied
schema. Schema references must be local; external resources are never downloaded.

The service adds JSON instructions before backend token budgeting and validates
output before returning it. **Decoding is not grammar-constrained.** A model may
produce invalid or truncated JSON; such output fails with HTTP 502 and code
`structured_output_invalid`, rather than returning invalid data. There are no
automatic retries. In streaming mode the JSON content is buffered until validation
passes; failure emits an SSE error without a successful finish. Tool calls remain
available as intermediate steps before the final JSON response.

`stream_options.include_usage` is accepted for all chat models. Counts are returned
when provided by the backend. Existing requests without `response_format` keep
their normal behavior. This feature does not implement `ai_task.evaluate` or the
`/v1/systemone` decision API.
