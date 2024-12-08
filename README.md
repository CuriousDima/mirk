# Mirk

**Mirk** is a library and a pipeline that combines classical Computer Vision (CV) models with Large Visual Models (LVMs) to provide detailed analysis and understanding of a video. The classical CV model handles initial processing and object detection, while the LVM generates rich, contextual interpretations of the visual content.

## Overview

Mirk works by:

1. Taking an input video
2. Using a CV model to detect objects of interest. Objects (classes) of interest are specified by the user
3. When a specified object is identified, triggering a VLM to generate detailed explanations about what is seen in the video, to reason about the detected object and its context based on the provided question

## Installation

```bash
pip install mirk
```

## Quick Start

Check out the [example](examples/one_shot.ipynb) to see how to use Mirk.

For your convenience, we provide a [bash script](examples/one_shot.sh) that downloads a sample video and runs the one-shot example:

```bash
cd examples
./one_shot.sh 
```

with the following output:

```bash
[download] Destination: input/selective_attention_test.mp4
...
[download] 100% of    2.63MiB in 00:00:00 at 5.80MiB/s
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to '.../mirk/mirk/models/yolo11n.pt'...
100%|███████| 5.35M/5.35M [00:00<00:00, 7.07MB/s]

video 1/1 (frame 1/2447) .../mirk/examples/input/selective_attention_test.mp4: 480x640 (no detections), 172.3ms
video 1/1 (frame 2/2447) .../mirk/examples/input/selective_attention_test.mp4: 480x640 (no detections), 145.1ms
video 1/1 (frame 3/2447) .../mirk/examples/input/selective_attention_test.mp4: 480x640 (no detections), 134.0ms
...
video 1/1 (frame 361/2447) .../mirk/examples/input/selective_attention_test.mp4: 480x640 5 persons, 160.9ms
Found person in frame 360 with confidence 0.88
Saved frame to: output/detected_person_frame_360.jpg

Question: What are the people doing in the image?
Answer: The people in the image are playing with basketballs, passing them to each other. There is a group of individuals, and some are walking while others are engaged in the activity. It's a scene from a well-known experiment involving selective attention.
```

## Credentials

Mirk uses the following APIs:

- [YOLO](https://docs.ultralytics.com/quickstart/)
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction)

You need to set up your own credentials for OpenAI API. See [.env.example](.env.example) file.  
You don't need to set up credentials for YOLO.
