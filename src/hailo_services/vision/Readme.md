# Object detection

YOLO/NMS decoding, image preprocessing and detector ownership in `vision.py`; Frigate ZeroMQ framing in `vision_zmq.py`. Detection has a separate owner queue while sharing physical Hailo resources. Frigate conversational/VLM planning lives in `assistants/frigate/`.
