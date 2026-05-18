#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/ffmpeg"
INSTALL="$ROOT/build_ffmpeg_install"

rm -rf "$BUILD" "$INSTALL"
mkdir -p "$BUILD"

tar xf "$ROOT/ffmpeg/ffmpeg-snapshot.tar.bz2" -C "$BUILD" --strip-components=1

export FFMPEG_PATCH_DIR="$ROOT/ffmpeg_patch"
export FFMPEG_INSTALL_DIR="$BUILD"
bash "$FFMPEG_PATCH_DIR/patch.sh"

cd "$BUILD"
./configure \
  --prefix="$INSTALL" \
  --enable-shared \
  --disable-static \
  --disable-programs \
  --disable-doc \
  --disable-debug \
  --enable-avcodec \
  --enable-avformat \
  --enable-avutil \
  --enable-swresample \
  --enable-swscale \
  --enable-protocol=file \
  --enable-demuxer=mov \
  --enable-demuxer=matroska \
  --enable-demuxer=h264 \
  --enable-demuxer=hevc \
  --enable-parser=h264 \
  --enable-parser=hevc \
  --enable-decoder=h264 \
  --enable-decoder=hevc

make -j"$(nproc)"
make install

mkdir -p "$ROOT/src/compressed_video_preinfer/libs"
cp "$INSTALL"/lib/libavcodec.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libavformat.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libavutil.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libswresample.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libswscale.so* "$ROOT/src/compressed_video_preinfer/libs/"
