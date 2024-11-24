import sounddevice as sd
import numpy as np

# Parameters
SAMPLERATE = 44100  # Sampling rate (Hz)
CHUNK_SIZE = 1024   # Number of audio frames per chunk
LOUDNESS_THRESHOLD = 0.5  # Adjust based on sensitivity (RMS value, 0 to 1)
DEVICE_INDEX = 0  # Replace with your microphone's device index, or use None for default

def audio_callback(indata, frames, time, status):
    """Callback to process audio input."""
    if status:
        print(f"Audio stream status: {status}")
    # Calculate the RMS (Root Mean Square) value of the audio chunk
    rms = np.sqrt(np.mean(indata**2))
    
    # Print the RMS value (optional for debugging)
    print(f"RMS: {rms:.4f}")
    
    # Detect loud sounds
    if rms > LOUDNESS_THRESHOLD:
        print("Loud sound detected!")

def main():
    """Main function to capture audio and detect loud sounds."""
    print("Starting loud sound detection. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, device=DEVICE_INDEX, callback=audio_callback):
            # Keep the script running
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopping loud sound detection.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
