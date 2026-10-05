# Qwen2-VL Home Assistant context benchmark

`scripts/benchmark-qwen-ha.py` measures where Qwen2-VL stops being reliable on realistic Home Assistant text and tool-calling tasks as the prompt grows.

The benchmark deliberately selects `Qwen2-VL-2B-Instruct` in every request and does **not** use the production `Static Context:` Home Assistant envelope. This prevents deterministic HA routing from answering the request before Qwen runs. The benchmark therefore measures Qwen itself, including the VLM prompt adapter and tokenizer budget.

## What is tested

The built-in scenarios mirror common Home Assistant requests:

- set light brightness (`light__HassLightSet`)
- set climate target temperature (`climate__HassClimateSetTemperature`)
- set a cover position (`intent__HassSetPosition`)
- query a live state (`homeassistant__GetLiveContext`)
- start a vacuum (`vacuum__HassVacuumStart`)
- read a current temperature from supplied HA-like state context without a tool call

The fixture contains representative rooms, devices and states similar to a real HA installation. It is not tied to a specific installation. Tool-call validation rejects unknown areas or device names, wrong tools, schema violations, wrong target values and extra unexpected arguments.

For tool scenarios the default request contains the expected tool plus two plausible distractor tools. `--all-tools` sends the full built-in tool set and stresses tool selection more aggressively.

## Run

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY
```

The API key can also be provided through `HAILO_API_KEY`; an explicit `--api-key` takes precedence.

The default requested input-token levels are:

```text
850,1000,1150,1300,1450,1550,1650,1725,1800,1900
```

These are sizing targets. Qwen's server-side tokenizer is authoritative. The report records both `input_tokens` and, when the service exposes it, `input_budget_tokens`. The latter includes the service's safety margin and is the best value to compare with the configured VLM input limit.

For a more statistically useful run repeat every point several times:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --repeats 5
```

To stress Qwen with all known benchmark tools:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --all-tools \
  --repeats 3
```

Run only selected scenarios:

```bash
python3 scripts/benchmark-qwen-ha.py \
  --url http://HOST:8090 \
  --api-key YOUR_API_KEY \
  --tasks light_brightness,state_temperature \
  --targets 900,1100,1300,1450,1550,1650,1725
```

## Output

The console prints one line per run with requested token level, measured token count, PASS/FAIL, duration and the validation reason. The output directory contains:

- `results.json` with the complete request, response, metrics and validation data
- `results.csv` for plotting correctness and latency against input-token count

The summary reports the highest measured input-token count that still produced a correct answer for each scenario and the first requested level that failed.

A failure is not limited to HTTP/context errors. It also includes a wrong tool, invalid JSON/tool arguments, schema violations, wrong values, or hallucinated room/device names. This makes it possible to find the **quality boundary before the hard context limit**, which is the relevant threshold for deciding whether a request should stay on the Qwen fast path or fall back to Gemma.
