// Audio Player Worklet Processor - Based on Node.js reference
// Handles proper buffering for streaming audio playback

class ExpandableBuffer {
    constructor() {
        // Start with one second's worth of buffered audio capacity
        this.buffer = new Float32Array(24000); // 1 second at 24kHz
        this.readIndex = 0;
        this.writeIndex = 0;
        this.underflowedSamples = 0;
        this.isInitialBuffering = true;
        this.initialBufferLength = 12000; // 0.5 seconds initial buffer
        this.lastWriteTime = 0;
    }

    logTimeElapsedSinceLastWrite() {
        const now = Date.now();
        if (this.lastWriteTime !== 0) {
            const elapsed = now - this.lastWriteTime;
            // Removed verbose logging for cleaner console output
        }
        this.lastWriteTime = now;
    }

    write(samples) {
        this.logTimeElapsedSinceLastWrite();
        
        if (this.writeIndex + samples.length <= this.buffer.length) {
            // Enough space to append the new samples
        } else {
            // Not enough space...
            if (samples.length <= this.readIndex) {
                // Can shift samples to beginning of buffer
                const subarray = this.buffer.subarray(this.readIndex, this.writeIndex);
                // console.log(`Shifting audio buffer by ${this.readIndex} samples`);
                this.buffer.set(subarray);
            } else {
                // Need to grow the buffer capacity
                const newLength = (samples.length + this.writeIndex - this.readIndex) * 2;
                const newBuffer = new Float32Array(newLength);
                // console.log(`Expanding audio buffer from ${this.buffer.length} to ${newLength}`);
                newBuffer.set(this.buffer.subarray(this.readIndex, this.writeIndex));
                this.buffer = newBuffer;
            }
            this.writeIndex -= this.readIndex;
            this.readIndex = 0;
        }
        
        this.buffer.set(samples, this.writeIndex);
        this.writeIndex += samples.length;
        
        if (this.writeIndex - this.readIndex >= this.initialBufferLength) {
            // Filled initial buffer, can start playback
            if (this.isInitialBuffering) {
                // Removed verbose logging for cleaner console output
            }
            this.isInitialBuffering = false;
        }
    }

    read(destination) {
        let copyLength = 0;
        if (!this.isInitialBuffering) {
            // Only play after initial buffer is filled
            copyLength = Math.min(destination.length, this.writeIndex - this.readIndex);
        }
        
        if (copyLength > 0) {
            destination.set(this.buffer.subarray(this.readIndex, this.readIndex + copyLength));
            this.readIndex += copyLength;
            
            if (this.underflowedSamples > 0) {
                // Underflow recovered - removed verbose logging for cleaner console output
                this.underflowedSamples = 0;
            }
        }
        
        if (copyLength < destination.length) {
            // Buffer underflow - fill rest with silence
            destination.fill(0, copyLength);
            this.underflowedSamples += destination.length - copyLength;
        }
        
        if (copyLength === 0) {
            // Ran out of audio, restart buffering
            this.isInitialBuffering = true;
        }
    }

    clearBuffer() {
        this.readIndex = 0;
        this.writeIndex = 0;
        this.isInitialBuffering = true;
        console.log("Audio buffer cleared due to barge-in");
    }
}

class AudioPlayerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.playbackBuffer = new ExpandableBuffer();
        
        this.port.onmessage = (event) => {
            if (event.data.type === "audio") {
                this.playbackBuffer.write(event.data.audioData);
            } else if (event.data.type === "initial-buffer-length") {
                this.playbackBuffer.initialBufferLength = event.data.bufferLength;
                // Buffer length changed - removed verbose logging for cleaner console output
            } else if (event.data.type === "barge-in") {
                this.playbackBuffer.clearBuffer();
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0][0]; // Single channel output
        if (output) {
            this.playbackBuffer.read(output);
        }
        return true; // Continue processing
    }
}

registerProcessor("audio-player-processor", AudioPlayerProcessor); 