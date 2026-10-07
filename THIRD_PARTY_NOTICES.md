# Third-party software and model notices

`hailo-10h-services` source code is licensed under the **Apache License 2.0** unless a file explicitly states otherwise.

The project repository and source distribution do **not** contain or redistribute Hailo runtime binaries, Hailo HEF files, model weights, tokenizers downloaded from external model providers, Piper voice models, or other vendor model assets. Those components are installed separately from package repositories or downloaded at install/runtime from their respective upstream providers and remain subject to their own licenses and terms.

This inventory was reviewed on **2026-10-07** and is provided to help users identify relevant upstream components. The license files and notices supplied by the exact packages or assets a user installs remain authoritative.

## Python and system dependencies

Python dependencies are resolved separately through package repositories such as PyPI, and system libraries can be installed through the operating-system package manager. They are not relicensed by this project.

Notable direct dependencies include FastAPI, Uvicorn, NumPy, Pillow, SciPy, SoundFile, Wyoming, MCP, aiomqtt, pyzmq, PyYAML, Jinja2, jsonschema, HassIL, Home Assistant intents, RapidFuzz, safetensors and tokenizers. Optional/test dependencies include Piper TTS, LiteRT-LM, pytest, HTTPX, Ruff and Hypothesis.

These packages retain their upstream licenses. Examples relevant to this project include:

- `piper-tts` >= 1.3: GPL-3.0-or-later
- LiteRT-LM: Apache-2.0
- HassIL and Home Assistant intents: Apache-2.0
- Wyoming: MIT
- SoundFile: BSD-3-Clause; the separately installed `libsndfile` library has its own LGPL terms
- pyzmq: BSD-3-Clause; ZeroMQ/libzmq has its own MPL terms where applicable

Installing or using a separately licensed dependency does not change the Apache-2.0 license of this repository's own source code. If someone creates and redistributes a combined binary image or appliance containing third-party components, the licenses of the components actually included in that distribution must also be followed.

## Hailo runtime and HEF files

HailoRT and related vendor runtime packages are prerequisites supplied separately from this repository. The project can use locally installed Hailo packages but does not distribute them as part of its source release.

Likewise, Hailo HEF files referenced by the model catalog are not stored in this repository. The service downloads configured model assets from external Hailo or other upstream sources when required and caches them locally for the user.

The repository's model catalog is therefore a list of identifiers, metadata and download locations; it is **not** a license grant for the referenced artifacts.

Upstream references:

- <https://github.com/hailo-ai/hailort>
- <https://github.com/hailo-ai/hailo_model_zoo>
- <https://github.com/hailo-ai/hailo_model_zoo_genai>

## Models and tokenizers

The service supports or references model families including Qwen, Whisper, MiniLM, Gemma, Llama, DeepSeek and YOLO. These models are obtained separately from their respective upstream providers or from Hailo-hosted compiled artifacts.

Their model licenses remain independent of the Apache-2.0 license of `hailo-10h-services`. In particular:

- Llama 3.2 uses the Meta Llama 3.2 Community License.
- Piper voice models use per-voice licenses; consult the voice model card supplied by the upstream provider.
- YOLO/Ultralytics-derived models can be subject to Ultralytics licensing terms, including AGPL-3.0 or a commercial license depending on the exact asset and use case.
- Qwen, Whisper, MiniLM, Gemma and DeepSeek assets retain the terms attached to the exact upstream model revision that is downloaded.

Because these assets are not redistributed by this repository, users should review the license attached to the exact model they choose to download and deploy.

Useful upstream references include:

- Qwen: <https://huggingface.co/Qwen>
- Whisper: <https://github.com/openai/whisper>
- MiniLM: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Gemma: <https://ai.google.dev/gemma>
- Llama: <https://www.llama.com/llama-downloads/>
- DeepSeek: <https://huggingface.co/deepseek-ai>
- Piper: <https://github.com/OHF-Voice/piper1-gpl>
- Ultralytics licensing: <https://www.ultralytics.com/license>

## Distribution boundary

The intended project distribution consists of this repository's source code, configuration examples, scripts and documentation. External runtimes and models are acquired separately by the user.

If a third party chooses to publish a prebuilt wheel, container, appliance or operating-system image that embeds dependencies or model assets, that distributor is responsible for complying with the licenses and notices of everything included in that combined artifact.

## Trademarks

Hailo, Home Assistant, Frigate, Qwen, Gemma, Llama, Piper, Ultralytics, YOLO and other third-party names and marks belong to their respective owners. Their mention here identifies interoperability or upstream components and does not grant trademark rights.
