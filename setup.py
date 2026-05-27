from pathlib import Path

import numpy
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
ffmpeg_root = ROOT / "build_ffmpeg_install"

ext = Extension(
    "codec_video_prep.cv_reader_fast",
    sources=["native/cv_reader_fast.cpp"],
    include_dirs=[
        str(ffmpeg_root / "include"),
        numpy.get_include(),
    ],
    library_dirs=[],
    libraries=[],
    language="c++",
    extra_compile_args=["-std=c++11", "-O3"],
    extra_link_args=[
        f"-L{ffmpeg_root / 'lib'}",
        "-Wl,--no-as-needed",
        "-lavformat",
        "-lavcodec",
        "-lswscale",
        "-lavutil",
        "-Wl,--as-needed",
        "-Wl,-rpath,$ORIGIN/libs",
        "-Wl,-Bsymbolic",
    ],
)

packages = find_packages("src") + find_packages(".", include=["codec_selector", "codec_selector.*"])
package_dir = {
    "codec_video_prep": "src/codec_video_prep",
    "compressed_video_preinfer": "src/compressed_video_preinfer",
}

setup(
    name="codec-video-prep",
    version="0.2.4",
    description="Codec-aware video preprocessing for training and inference",
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23,<2.0",
        "opencv-python-headless<4.12",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "codec-video-prep=codec_video_prep.cli:main",
            "codec-video-prep-doctor=codec_video_prep.doctor:main",
            "cv-preinfer=compressed_video_preinfer.cli:main",
            "cv-preinfer-doctor=compressed_video_preinfer.doctor:main",
        ],
    },
    packages=packages,
    package_dir=package_dir,
    package_data={"codec_video_prep": ["libs/*.so*"]},
    ext_modules=[ext],
)
