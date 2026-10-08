# Resident worker processes

Worker supervision, IPC, streaming chunks, cancellation and structured errors. Hailo chat/STT and detector proxies share one worker; LiteRT and Piper each have their own resident worker. Native models remain owned by their worker executors.
