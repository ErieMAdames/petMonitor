import sounddevice as sd
import numpy as np

# Parameters
SAMPLERATE = 44100  # Sampling rate (Hz)
CHUNK_SIZE = 1024   # Number of audio frames per chunk
LOUDNESS_THRESHOLD = 0.5  # RMS value threshold for loud sounds
DEVICE_INDEX = 1  # Replace with your device index, or leave None for default
CHANNELS = 2  # Use 2 if your microphone supports only stereo

def audio_callback(indata, frames, time, status):
    """Callback to process audio input."""
    if status:
        print(f"Audio stream status: {status}")
    # Calculate the RMS for the first channel only
    rms = np.sqrt(np.mean(indata[:, 0]**2))
    
    print(f"RMS: {rms:.4f}")
    if rms > LOUDNESS_THRESHOLD:
        print("Loud sound detected!")

def main():
    """Main function for loud sound detection."""
    print("Starting loud sound detection. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, device=DEVICE_INDEX, callback=audio_callback):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopping loud sound detection.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
