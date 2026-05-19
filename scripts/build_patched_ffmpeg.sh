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

mkdir -p "$INSTALL/include" "$INSTALL/lib"
for component in libavcodec libavformat libavutil libswresample libswscale; do
  mkdir -p "$INSTALL/include/$component"
  cp "$component"/*.h "$INSTALL/include/$component/"
  cp -P "$component"/lib*.so* "$INSTALL/lib/"
done

mkdir -p "$ROOT/src/codec_video_prep/libs"
cp "$INSTALL"/lib/libavcodec.so* "$ROOT/src/codec_video_prep/libs/"
cp "$INSTALL"/lib/libavformat.so* "$ROOT/src/codec_video_prep/libs/"
cp "$INSTALL"/lib/libavutil.so* "$ROOT/src/codec_video_prep/libs/"
cp "$INSTALL"/lib/libswresample.so* "$ROOT/src/codec_video_prep/libs/"
cp "$INSTALL"/lib/libswscale.so* "$ROOT/src/codec_video_prep/libs/"
