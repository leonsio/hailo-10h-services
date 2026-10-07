# Third-party notices and model licensing

`hailo-10h-services` source code is licensed under **GNU GPL-3.0-or-later** unless a file explicitly states otherwise. The project license does **not** relicense third-party libraries, Hailo runtime packages, HEF files, model weights, tokenizers, Piper voice models, operating-system packages, or other downloaded assets.

This inventory was reviewed on **2026-10-07**. Dependency version ranges and externally downloaded models can change, so the license files and notices shipped with the exact artifacts you distribute remain authoritative. This document is an engineering inventory, not legal advice.

## Runtime and direct Python dependencies

| Component | Purpose in this project | Upstream license / note |
| --- | --- | --- |
| FastAPI | HTTP API | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| python-multipart | Multipart uploads | Apache-2.0 |
| NumPy | Numerical arrays | BSD-3-Clause; binary wheels can contain separately licensed bundled components |
| Pillow | Image handling | MIT-CMU |
| SciPy | Signal processing / numerical routines | BSD-3-Clause |
| SoundFile | Audio file I/O | BSD-3-Clause; its `libsndfile` dependency is LGPL-2.1-or-later |
| Wyoming | Home Assistant voice protocol | MIT |
| MCP Python SDK | Model Context Protocol | MIT |
| aiomqtt | MQTT client | BSD-3-Clause; commonly uses Eclipse Paho MQTT, which is dual EPL-2.0 / EDL-1.0 |
| pyzmq | ZeroMQ Python bindings | BSD-3-Clause; bundled/system `libzmq` is MPL-2.0 |
| PyYAML | YAML parsing | MIT |
| Jinja2 | Templates | BSD-3-Clause |
| jsonschema | JSON Schema validation | MIT |
| HassIL | Home Assistant intent matching | Apache-2.0 |
| home-assistant-intents | Home Assistant intent data | Apache-2.0 |
| RapidFuzz | Fuzzy matching | MIT |
| safetensors | Model weight format | Apache-2.0 |
| tokenizers | Tokenizer runtime | Apache-2.0 |
| `piper-tts` >= 1.3 | Optional/direct CPU TTS backend; installed by the supplied Dockerfile | GPL-3.0-or-later |
| LiteRT-LM | Optional CPU LLM runtime | Apache-2.0 |

The project directly imports the current GPL Piper package when Piper is enabled. Choosing GPL-3.0-or-later for this repository is compatible with that integration; it does not change Piper's own copyright or license notices.

### Development/test dependencies

| Component | Upstream license |
| --- | --- |
| pytest | MIT |
| HTTPX | BSD-3-Clause |
| Ruff | MIT |
| Hypothesis | MPL-2.0 |

## Hailo software and binaries

The repository expects Hailo runtime packages to be supplied or installed separately. The Docker build copies a local Hailo DEB and Python wheel from `deploy/vendor/`; those vendor binaries are **not** relicensed by this repository.

Hailo currently documents:

- `libhailort`, `pyhailort`, and `hailortcli`: **MIT**
- `hailonet` GStreamer plugin: **LGPL-2.1-or-later**
- Hailo Model Zoo source/tooling: **MIT**
- Hailo GenAI Model Zoo source/tooling: **MIT**

Upstream references:

- <https://github.com/hailo-ai/hailort>
- <https://github.com/hailo-ai/hailo_model_zoo>
- <https://github.com/hailo-ai/hailo_model_zoo_genai>

A Model Zoo repository license should not be treated as a blanket relicense of every upstream model or compiled HEF. Preserve any notices or terms supplied with the exact Hailo download you redistribute.

## Models, HEFs, tokenizers, and voices

The model catalog contains download references. A catalog entry is **not** a grant of rights to the referenced asset. Models downloaded at runtime remain under the terms of their upstream model authors and/or the distributor of the compiled artifact.

| Catalog family / asset | Upstream license information to preserve |
| --- | --- |
| Qwen2 / Qwen2.5 / Qwen3 text models | The referenced upstream Qwen model families are generally Apache-2.0; check the exact model card and any Hailo artifact terms |
| Qwen2-VL-2B-Instruct | Apache-2.0 upstream model |
| Qwen3-VL-2B-Instruct | Apache-2.0 upstream model; check the exact revision used |
| DeepSeek-R1-Distill-Qwen-1.5B | MIT for the DeepSeek repository and model weights; derived from Apache-2.0 Qwen 2.5 |
| Llama3.2-1B-Instruct | Meta **Llama 3.2 Community License**, not GPL/MIT/Apache; redistribution has its own attribution and notice requirements |
| Whisper Tiny/Base/Small | MIT upstream code and model weights |
| all-MiniLM-L6-v2 tokenizer/weights | Apache-2.0 |
| `cstr/all-MiniLM-L6-v2-hailo10h` community HEF | Apache-2.0 as currently declared by that repository; HailoRT compatibility is separate |
| Gemma 4 E2B / LiteRT community package | Gemma 4 is currently published under Apache-2.0; preserve the exact model/package notices that accompany the downloaded asset |
| Piper voices | License is **per voice**. Review the voice `MODEL_CARD`; some voices have restrictive terms |
| YOLOv8 / YOLO11 / YOLO26 HEFs | Hailo Model Zoo tooling is MIT, but this does not by itself settle the license of the underlying trained model. Ultralytics states its trained YOLO models are AGPL-3.0 by default unless covered by an Enterprise license. Verify the provenance and license of the exact HEF before redistribution or deployment |

Useful upstream references:

- Qwen2-VL-2B-Instruct: <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct>
- Qwen2.5-Coder-1.5B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>
- Qwen3 family: <https://huggingface.co/Qwen/Qwen3-1.7B>
- DeepSeek-R1-Distill-Qwen-1.5B: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>
- Llama 3.2 license: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt>
- Whisper: <https://github.com/openai/whisper>
- all-MiniLM-L6-v2: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Gemma model documentation: <https://ai.google.dev/gemma>
- Piper: <https://github.com/OHF-Voice/piper1-gpl>
- Piper voice licensing guidance: <https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md>
- Ultralytics licensing: <https://www.ultralytics.com/license>

### YOLO / AGPL caution

The repository's GPL-3.0-or-later license does not waive or replace rights attached to a separately downloaded YOLO model. If a selected HEF is derived from an Ultralytics model covered by AGPL-3.0, additional AGPL obligations can apply to that deployment or distribution. Do not infer that a Hailo-hosted `.hef` download is automatically MIT merely because the Hailo Model Zoo source repository is MIT.

For commercial or closed deployments, determine the provenance of the exact YOLO HEF and obtain an appropriate upstream license where required.

## Piper voice caution

`piper-tts` itself is GPL-3.0-or-later, but Piper explicitly notes that individual voice models can use different and sometimes restrictive licenses. The service only loads locally provisioned ONNX/JSON voice pairs; anyone distributing such voices must preserve and comply with the corresponding voice `MODEL_CARD` and license.

## Binary/container distribution checklist

When publishing a wheel, Docker image, appliance image, or other binary bundle built from this repository:

1. Include this project's `LICENSE` and `THIRD_PARTY_NOTICES.md`.
2. Make the corresponding source for GPL-covered project code available as required by GPL-3.0-or-later.
3. Preserve the license/notice files supplied by all packaged Python wheels and native libraries.
4. Preserve LGPL/MPL/EPL/EDL notices for components actually present in the distributed image.
5. Do not bundle Hailo vendor packages unless their distribution terms permit it.
6. Do not bundle HEFs, model weights, tokenizers, or Piper voices without checking the exact asset's license.
7. For Llama 3.2 redistribution, comply with the Meta Llama 3.2 Community License and its attribution/notice requirements.
8. For Ultralytics-derived YOLO assets, review the current AGPL-3.0/Enterprise terms before distribution or network deployment.

## No trademark grant

Names such as Hailo, Home Assistant, Frigate, Qwen, Gemma, Llama, Piper, Ultralytics, YOLO, and other third-party marks belong to their respective owners. Their mention here identifies interoperability or upstream components and does not grant trademark rights.
