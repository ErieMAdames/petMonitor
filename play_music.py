from playsound import playsound
import threading
import time

MUSIC_FILE = "relaxing.mp3"  # Path to your music file

def play_music():
    """Plays the relaxing music in a separate thread."""
    threading.Thread(target=playsound, args=(MUSIC_FILE,), daemon=True).start()

if __name__ == "__main__":
    try:
        play_music()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
