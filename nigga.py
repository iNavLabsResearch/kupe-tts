import time
import soundfile as sf

from tts_runner import create_runner

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Change to "standard" to use the original OmniVoice PyTorch model
MODEL_TYPE = "triton"

# Reference data for the voice clone
ref_audio_file = "darshan.wav"
ref_text_content = (
    "Thank you for calling support. I completely understand the issue you're facing. "
    "में आ गई है और हमारी टीम अभी इसे चेक कर रही है। "
    "हम जल्द से जल्द इसका समाधान करने की پوری کوشیش करेंگे."
)

# Text chunks to synthesize
chunks = [
    "The temperature was -3.5 degrees and visibility was 0 km.",
    "Pai is approximately 3.14159.",
    "મારી પાસે 3 પુસ્તકો અને 10 પેન છે.",
    "There are 1000000 bytes in a megabyte.",
    "मेरे पास 2 किताबें और 5.85 पेन हैं।",
]

# ---------------------------------------------------------------------------
# Create runner and generate
# ---------------------------------------------------------------------------

runner = create_runner(model_type=MODEL_TYPE)

total_start_time = time.time()

for i, chunk_text in enumerate(chunks):
    start_time = time.time()

    # Generate audio for the chunk using voice cloning
    result = runner.generate_voice_clone(
        text=chunk_text,
        ref_audio=ref_audio_file,
        ref_text=ref_text_content,
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"Chunk {i} processed in {duration:.4f} seconds: '{chunk_text}'")

    # Save each chunk to a unique filename
    filename = f"darshan_chunk_{i}.wav"
    sf.write(filename, result.audio, result.sample_rate)
    print(f"Saved to {filename}\n")

total_end_time = time.time()
print(f"Total time taken for all chunks: {total_end_time - total_start_time:.4f} seconds")
