#! /bin/bash

# Create an input directory if it doesn't exist
mkdir -p "input"

# Download the video file to the input directory
VIDEO_NAME="selective_attention_test.mp4"
yt-dlp --no-audio -f "bestvideo[ext=mp4]" -o "input/${VIDEO_NAME}" "https://www.youtube.com/watch?v=vJG698U2Mvo"

# Run the script with the following arguments:
# --video-path: Path to the video file
# --question: Question to ask about the image
# --output-dir: Directory to save output files
python one_shot.py \
    --video-path "input/${VIDEO_NAME}" \
    --question "What are the people doing in the image?" \
    --output-dir "output"
