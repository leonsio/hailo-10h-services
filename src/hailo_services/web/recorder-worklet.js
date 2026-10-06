// PCM capture; output remains silent so the microphone is never played back live.
/** Capture mono PCM blocks without routing microphone sound to the speakers. */
class PCMRecorder extends AudioWorkletProcessor {
  /**
   * Average input channels and transfer the mono block to the recorder UI.
   * @param {Float32Array[][]} inputs Audio-worklet input buses and channels.
   * @returns {boolean} True to keep the processor active.
   */
  process(inputs) {
    const channels = inputs[0];
    if (channels?.length && channels[0].length) {
      const mono = new Float32Array(channels[0].length);
      for (const channel of channels) for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / channels.length;
      this.port.postMessage(mono, [mono.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-recorder", PCMRecorder);
