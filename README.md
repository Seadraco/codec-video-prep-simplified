# Compressed Video Reader

The Compressed Video Reader is designed to read motion vectors and residuals from H.264/H.265 encoded videos.

## Installation

To install the reader, you can run the installation script located in the project root:

```shell
bash install.sh
```

The script will perform the following tasks:

1. Download the source code of FFmpeg
2. Apply patches to the source code
3. Configure and compile the FFmpeg package
4. Build and install the reader

To test if the reader has been successfully installed, run the following command:

```bash
# Test if the reader is installed successfully.
cv_reader -h || echo "Installation failed!"
```

## Python API

### Basic Usage (Load All Frames)

```python
import cv_reader
video_frames = cv_reader.read_video(video_path=path_to_video, with_residual=True)
```

### Streaming API (Memory Efficient for Long Videos)

For long videos, use `read_video_cb` to process frames one by one without loading all into memory:

```python
import cv_reader

def process_frame(frame_dict):
    """Callback function called for each decoded frame.
    
    Args:
        frame_dict: Dictionary containing frame data:
            - 'frame_idx': Frame index
            - 'pict_type': Frame type ('I', 'P', 'B')
            - 'motion_vector': Motion vectors (H/4, W/4, 4) int32
            - 'motion_energy': Raw motion energy (optional)
            - 'motion_energy_median': Global median compensated energy (optional)
            - 'residual_y': Residual Y plane (optional)
    
    Returns:
        True to continue, False to stop decoding
    """
    frame_idx = frame_dict['frame_idx']
    pict_type = frame_dict['pict_type']
    mv = frame_dict['motion_vector']
    
    # Process frame data...
    print(f"Frame {frame_idx} ({pict_type}): MV shape {mv.shape}")
    
    return True  # Continue to next frame

# Stream video without loading all frames into memory
cv_reader.read_video_cb(
    path_to_video,
    callback=process_frame,
    without_residual=0,      # 0=with residual, 1=without
    max_frames=0,            # 0=no limit
    frame_ids=None,          # None=all frames, or list [0, 10, 20]
    seek_to_frame=-1,        # -1=from start, or frame index to seek
    decode_len=0,            # 0=to end, or number of frames to decode
    residual_rgb=1           # 1=RGB residual, 0=YUV residual
)
```

## CLI Interface

You can use the following command to extract motion vectors and residuals from a compressed video:

```text
$ cv_reader -h
usage: Compressed Video Reader [-h] video output

positional arguments:
  video       Path to h.264/h.265 video file
  output      Path to save extracted motion vectors and residuals

optional arguments:
  -h, --help  show this help message and exit
```

To run the extraction process on the example video, execute the following command:

```bash
python debug_vis_mvres.py --video ../test_videos/h264_sample.mp4 --num_frames 16 --out_dir ./h264_debug
python debug_vis_mvres.py --video ../test_videos/h265_sample.mp4 --num_frames 16 --out_dir ./h265_debug
```