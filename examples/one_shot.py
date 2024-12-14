#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

sys.path.append("../mirk")

from mirk.providers_cv import YOLOProvider
from mirk.providers_vlm import OpenAIVLMProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--video-path", type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "-c",
        "--target-class",
        type=str,
        default="person",
        help="Target class to detect",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="What are the people doing in the image?",
        help="Question to ask about the image",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="examples/output",
        help="Directory to save output files",
    )
    args = parser.parse_args()

    if not args.video_path:
        parser.error("--video-path argument is required")

    return args


def main():
    args = parse_args()

    # Initialize provider and use video path from arguments
    # Supported YOLO classes can be found at:
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    cv_yolo = YOLOProvider()

    # Detect an object in video - get first detection that meets our confidence threshold
    # You can use for loop to get all detections that meet our confidence threshold.
    try:
        frame_num, confidence = next(
            cv_yolo.detect(
                source=args.video_path,
                target_class=args.target_class,
                conf_threshold=0.8,
            )
        )
        print(
            f"Found {args.target_class} in frame {frame_num} with confidence {confidence:.2f}"
        )

        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the detected frame
        output_path = output_dir / f"detected_{args.target_class}_frame_{frame_num}.jpg"
        cv_yolo.save_frame(args.video_path, frame_num, str(output_path))
        print(f"Saved frame to: {output_path}")

        # Ask about the saved frame
        # OpenAIVLMProvider will use OPENAI_API_KEY environment variable.
        # See .env.example file. Create your own .env file to use your own API key.
        vlm_openai = OpenAIVLMProvider(model="gpt-4o")
        answer = vlm_openai.ask_about_image(str(output_path), args.question)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")

    except StopIteration:
        print(f"Target object '{args.target_class}' not found in video")


if __name__ == "__main__":
    main()
