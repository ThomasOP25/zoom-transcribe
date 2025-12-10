import whisper
import os
import subprocess

def make_transcript(model, filename):
    result = model.transcribe(f"{filename}.mp3")

    os.makedirs("zoom_transcribe", exist_ok=True)
    transcript_path = f"zoom_transcribe/{filename}.txt"
    with open(transcript_path, "w") as file:
        
        file.write(result["text"])

    return result["text"]

def summarize_with_deepseek(text, filename):
    try:
        prompt_text = f"summarize this text:\n\n{text}"
        # Run ollama with input from stdin
        process = subprocess.run(
            ["ollama", "run", "deepseek-r1"],
            input=prompt_text,
            capture_output=True,
            text=True,
            check=True,
        )
        raw_output = process.stdout.strip()

        # Split on the chain-of-thought ending marker
        if "done thinking." in raw_output:
            summary = raw_output.split("done thinking.")[-1].strip()
        else:
            summary = raw_output

        summary_path = f"zoom_transcribe/{filename}_summary.txt"
        with open(summary_path, "w") as file:
            file.write(summary)

        return summary
    except subprocess.CalledProcessError as e:
        print("Error during summarization:", e)
        print("stderr:", e.stderr)
        return None

def main():
    print("\nLoading OpenAI Whisper...\n")
    model = whisper.load_model("medium")
    filename = input("\nEnter filename (without .mp3): \n")
    print("\nTranscribing speech to text...\n")
    transcription = make_transcript(model, filename)
    print("\n--- Transcription ---\n")
    print(transcription)

    print("\n--- Summary (Deepseek) ---\n")
    print("\nSummarizing...\n")
    summary = summarize_with_deepseek(transcription, filename)
    if summary:
        print(summary)
    else:
        print("Failed to get summary.")

if __name__ == "__main__":
    main()
