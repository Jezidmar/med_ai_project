import librosa
import wave
import os
import time
import whisper
import torch
import pyaudio
from execute_data_collection import decode


# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
(get_speech_ts, _, _, _, _) = utils

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample rate for VAD (16000 Hz for Silero)
CHUNK = 1000  # Buffer size for recording
MAX_SILENCE_DURATION = 10  # Max silence allowed before stopping (in seconds)
VAD_WINDOW = 5  # Number of audio chunks to accumulate for VAD decision
# Configure the Gemini API


p = pyaudio.PyAudio()


# Function to save a buffer to a temporary WAV file
def save_chunk_to_file(buffer, filename="temp_chunk.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(buffer)


# Function to check VAD using librosa for audio loading
def is_speech_using_librosa(filename="temp_chunk.wav"):
    y, sr = librosa.load(filename, sr=16000)
    speech_segments = get_speech_ts(y, model, sampling_rate=sr)
    return len(speech_segments) > 0


def load_entire_file_and_transcribe(filename="output.wav"):
    y, _ = librosa.load(filename, sr=16000)
    model = "medium.en"

    audio_model = whisper.load_model(model)
    result = audio_model.transcribe(y, fp16=torch.cuda.is_available())
    text = result["text"].strip()
    return text


# Modify record_audio function to process 2-second chunks
def record_audio(file_name):
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print("Recording...")

    recorded_data = []
    last_speech_time = time.time()
    vad_buffer = b""  # Buffer to accumulate audio data for better VAD detection
    chunk_counter = 0

    while True:
        data = stream.read(CHUNK)
        vad_buffer += data
        chunk_counter += 1

        # If we've accumulated 2 seconds (32 chunks), process the chunk
        if chunk_counter >= RATE * 2 // CHUNK:
            print("Processing 2-second chunk...")
            save_chunk_to_file(vad_buffer, filename="temp_chunk.wav")
            if is_speech_using_librosa("temp_chunk.wav"):
                last_speech_time = time.time()
                print("Speech detected in 2-second chunk.")
            else:
                print("No speech detected in 2-second chunk.")

            vad_buffer = b""  # Reset buffer after processing
            chunk_counter = 0

        recorded_data.append(data)

        # Check if there has been silence for too long
        if time.time() - last_speech_time > MAX_SILENCE_DURATION:
            print("No speech detected for 10 seconds. Stopping recording.")
            break

    # Stop the stream
    stream.stop_stream()
    stream.close()

    # Save the entire recorded audio to a file
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(recorded_data))

    print(f"Recording saved to {file_name}")


# Main function to start the recording process
def main():
    output_file = "output.wav"
    record_audio(output_file)

    # Cleanup temporary chunk file after processing
    if os.path.exists("temp_chunk.wav"):
        os.remove("temp_chunk.wav")
    input_text = load_entire_file_and_transcribe()
    print(f"Recorded speech:{input_text}")
    decode(input_text)
