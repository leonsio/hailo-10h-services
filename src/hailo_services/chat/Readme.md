# Resident chat backends

`backend_hailo.py` owns SHARED native models; `backend_litert.py` owns CPU LiteRT-LM. Role adapters use `chat_common.py` for Hailo prompts, native budgeting and tool validation. Scheduling belongs in `runtime/`, domain decisions in `assistants/`.
