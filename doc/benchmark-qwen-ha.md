# Qwen2-VL Home Assistant context benchmark

`scripts/benchmark-qwen-ha.py` measures where Qwen2-VL stops being reliable on realistic Home Assistant text and tool-calling tasks as the prompt grows.

Every request explicitly selects `Qwen2-VL-2B-Instruct` and deliberately avoids the exact production `Static Context:` Home Assistant envelope. Deterministic HA routing therefore cannot answer the benchmark instead of Qwen.

## What is tested

The built-in scenarios cover:

- light brightness with `light__HassLightSet`
- climate target temperature with `climate__HassClimateSetTemperature`
- cover position with `intent__HassSetPosition`
- live-state lookup with `homeassistant__GetLiveContext`
- vacuum start with `vacuum__HassVacuumStart`
- direct reading of a supplied current temperature without a tool call

The fixture uses representative Home Assistant entities and states, but production logic is not tied to these names.

## Important: the default run is a baseline

The default `focused` mode sends only the tool required by the current task and only the relevant entities. This isolates Qwen's context/reasoning boundary from tool-selection complexity.

The requested input-token targets are:

```text
400,550,700,850,1000,1150,1300,1450,1600,1750
```

They are sizing targets only. `input_budget_tokens` reported by the service is authoritative when available.

After the focused baseline you can separately increase tool complexity:

```bash
--tool-mode distractors
--tool-mode all
```

`distractors` adds two plausible competing tools. `all` sends the full built-in benchmark tool set. Note that the service's MiniLM tool retrieval may still reduce that set before Qwen sees it; debug logs show the final tool list.

`--full-catalogue` adds the complete base entity catalogue before the synthetic archive context. Use it only after the focused baseline.

## Recommended first run

Start with one scenario and low context levels:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --tasks light_brightness \
  --targets 300,400,500,600,700 \
  --repeats 1
```

The API key may also be supplied through `HAILO_API_KEY`.

If the baseline is stable, run all scenarios:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --repeats 3
```

Then compare tool-selection complexity:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --tool-mode distractors \
  --repeats 3
```

## Failure classification

Model failures are benchmark results and do not make the script exit with status 1. Only preflight/execution failures stop the benchmark.

The console now distinguishes, among others:

- `model output: required tool call missing`
- `model output: unavailable function`
- `model output: invalid tool arguments`
- `model output: invalid tool JSON`
- `input token limit`
- semantic validation failures such as `wrong target/value` or `hallucinated area/name`

For successful HTTP responses, failed tool validation also prints the generated tool and arguments. HTTP failures print the server error text. This matters because the service can accept the input-token budget, run Qwen, and only then reject malformed model-generated tool output.

## Output

The output directory contains:

- `results.json` with complete requests, successful responses, HTTP/server errors, metrics and validation data
- `results.csv` with token counts, latency, HTTP status, validation reason and compact failure detail

The summary reports the highest measured input-token count that produced a correct result, the first failed target and a count of failure reasons per scenario.
