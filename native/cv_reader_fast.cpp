/*
 * Fast video decoder Python C extension with bitcost export.
 *
 * Stripped-down version of cv_reader that does only the essentials:
 *  - open / find stream / init decoder
 *  - decode loop with skip_loop_filter + skip_idct
 *  - optional H.264/HEVC bitcost export from frame->opaque_ref
 *  - returns a lightweight list of dicts with frame metadata
 *
 * Link against the patched FFmpeg static libs with -Wl,-Bsymbolic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
}

#include <Python.h>
#include <numpy/arrayobject.h>
#include <unordered_map>
#include <vector>

/* ------------------------------------------------------------------ */
/* H264MbBitCostMap – must match ffmpeg_patch/h264_cabac.c exactly   */
/* ------------------------------------------------------------------ */
#define H264_MB_BIT_COST_MAP_MAGIC   0x4D424354U   /* MKBETAG('M','B','C','T') */
#define H264_MB_BIT_COST_MAP_VERSION 2
#define HEVC_CTU_BIT_COST_MAP_MAGIC  0x48435455U   /* MKBETAG('H','C','T','U') */
#define HEVC_CTU_BIT_COST_MAP_VERSION 2
#define CVR_TARGET_BITCOST_MAGIC     0x43565254U   /* MKBETAG('C','V','R','T') */

#pragma pack(push, 1)
typedef struct H264MbBitCostMap {
    uint32_t magic;
    int32_t  version;
    int32_t  mb_width;
    int32_t  mb_height;
    int32_t  mb_stride;
    int32_t  sub_width;
    int32_t  sub_height;
    int32_t  sub_stride;
} H264MbBitCostMap;

typedef struct HevcCtuBitCostMap {
    uint32_t magic;
    int32_t  version;
    int32_t  ctb_width;
    int32_t  ctb_height;
    int32_t  ctb_stride;
    int32_t  log2_ctb_size;
    int32_t  sub_width;
    int32_t  sub_height;
    int32_t  sub_stride;
} HevcCtuBitCostMap;
#pragma pack(pop)

typedef struct CvrTargetBitcostCtx {
    uint32_t magic;
    int enabled;
    int max_frame;
    const uint8_t *frame_bitmap;
    int tolerance;
} CvrTargetBitcostCtx;

static const int32_t *h264_mb_costs_ptr(const H264MbBitCostMap *map) {
    return (const int32_t *)((const uint8_t *)map + sizeof(H264MbBitCostMap));
}

static const float *h264_sub_costs_ptr(const H264MbBitCostMap *map) {
    return (const float *)(h264_mb_costs_ptr(map) + (size_t)map->mb_stride * map->mb_height);
}

static const int32_t *hevc_ctu_costs_ptr(const HevcCtuBitCostMap *map) {
    return (const int32_t *)((const uint8_t *)map + sizeof(HevcCtuBitCostMap));
}

static const float *hevc_sub_costs_ptr(const HevcCtuBitCostMap *map) {
    return (const float *)(hevc_ctu_costs_ptr(map) + (size_t)map->ctb_stride * map->ctb_height);
}

static int cvr_bitcost_diag_enabled_cpp()
{
    const char *v = getenv("CVR_BITCOST_DIAG");
    return v && v[0] && v[0] != '0';
}

static int cvr_bitcost_map_is_valid(AVFrame *frame)
{
    if (!frame->opaque_ref || !frame->opaque_ref->data)
        return 0;
    if ((size_t)frame->opaque_ref->size < sizeof(H264MbBitCostMap))
        return 0;
    const H264MbBitCostMap *map = (const H264MbBitCostMap *)frame->opaque_ref->data;
    if (map->magic != H264_MB_BIT_COST_MAP_MAGIC ||
        map->version != H264_MB_BIT_COST_MAP_VERSION) {
        const HevcCtuBitCostMap *hevc_map = (const HevcCtuBitCostMap *)frame->opaque_ref->data;
        if (hevc_map->magic != HEVC_CTU_BIT_COST_MAP_MAGIC ||
            hevc_map->version != HEVC_CTU_BIT_COST_MAP_VERSION)
            return 0;
        size_t ctu_count = (size_t)hevc_map->ctb_stride * hevc_map->ctb_height;
        size_t sub_count = (size_t)hevc_map->sub_stride * hevc_map->sub_height;
        size_t expect_sz = sizeof(HevcCtuBitCostMap) +
                           ctu_count * sizeof(int32_t) +
                           sub_count * sizeof(float);
        return (size_t)frame->opaque_ref->size >= expect_sz;
    } else {
        size_t mb_count = (size_t)map->mb_stride * map->mb_height;
        size_t sub_count = (size_t)map->sub_stride * map->sub_height;
        size_t expect_sz = sizeof(H264MbBitCostMap) +
                           mb_count * sizeof(int32_t) +
                           sub_count * sizeof(float);
        return (size_t)frame->opaque_ref->size >= expect_sz;
    }
}

static int cvr_bitcost_map_has_nonzero_mb(AVFrame *frame)
{
    if (!cvr_bitcost_map_is_valid(frame))
        return 0;
    const H264MbBitCostMap *h264_map = (const H264MbBitCostMap *)frame->opaque_ref->data;
    if (h264_map->magic == H264_MB_BIT_COST_MAP_MAGIC) {
        const int32_t *mb = h264_mb_costs_ptr(h264_map);
        size_t mb_count = (size_t)h264_map->mb_stride * h264_map->mb_height;
        for (size_t i = 0; i < mb_count; ++i) {
            if (mb[i] != 0)
                return 1;
        }
    } else {
        const HevcCtuBitCostMap *hevc_map = (const HevcCtuBitCostMap *)frame->opaque_ref->data;
        const int32_t *ctu = hevc_ctu_costs_ptr(hevc_map);
        size_t ctu_count = (size_t)hevc_map->ctb_stride * hevc_map->ctb_height;
        for (size_t i = 0; i < ctu_count; ++i) {
            if (ctu[i] != 0)
                return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: build a numpy array from existing data (copies)            */
/* ------------------------------------------------------------------ */
static PyObject *
make_numpy_copy(int ndim, const npy_intp *dims, int typenum,
                const void *src, size_t src_size)
{
    PyObject *arr = PyArray_SimpleNew(ndim, dims, typenum);
    if (!arr)
        return nullptr;
    void *dst = PyArray_DATA((PyArrayObject *)arr);
    memcpy(dst, src, src_size);
    return arr;
}

/* ------------------------------------------------------------------ */
/* Extract bitcost dict from frame->opaque_ref                         */
/* ------------------------------------------------------------------ */
static PyObject *
extract_h264_bitcost(AVFrame *frame)
{
    if (!frame->opaque_ref || !frame->opaque_ref->data)
        Py_RETURN_NONE;

    if ((size_t)frame->opaque_ref->size < sizeof(H264MbBitCostMap))
        Py_RETURN_NONE;

    const H264MbBitCostMap *map = (const H264MbBitCostMap *)frame->opaque_ref->data;

    if (map->magic != H264_MB_BIT_COST_MAP_MAGIC ||
        map->version != H264_MB_BIT_COST_MAP_VERSION)
        Py_RETURN_NONE;

    size_t mb_count   = (size_t)map->mb_stride   * map->mb_height;
    size_t sub_count  = (size_t)map->sub_stride  * map->sub_height;
    size_t expect_sz  = sizeof(H264MbBitCostMap)
                        + mb_count  * sizeof(int32_t)
                        + sub_count * sizeof(float);

    if ((size_t)frame->opaque_ref->size < expect_sz)
        Py_RETURN_NONE;

    const int32_t *mb_costs  = h264_mb_costs_ptr(map);
    const float   *sub_costs = h264_sub_costs_ptr(map);

    npy_intp mb_dims[2]  = {map->mb_height,  map->mb_stride};
    npy_intp sub_dims[2] = {map->sub_height, map->sub_stride};

    PyObject *mb_arr  = make_numpy_copy(2, mb_dims,  NPY_INT32,
                                        mb_costs,  mb_count * sizeof(int32_t));
    PyObject *sub_arr = make_numpy_copy(2, sub_dims, NPY_FLOAT32,
                                        sub_costs, sub_count * sizeof(float));
    if (!mb_arr || !sub_arr) {
        Py_XDECREF(mb_arr);
        Py_XDECREF(sub_arr);
        return nullptr;
    }

    PyObject *dict = PyDict_New();
    if (!dict) {
        Py_DECREF(mb_arr);
        Py_DECREF(sub_arr);
        return nullptr;
    }

    PyDict_SetItemString(dict, "mb_bit_cost",  mb_arr);
    PyDict_SetItemString(dict, "sub_mb_bit_cost", sub_arr);
    PyDict_SetItemString(dict, "mb_width",     PyLong_FromLong(map->mb_width));
    PyDict_SetItemString(dict, "mb_height",    PyLong_FromLong(map->mb_height));
    PyDict_SetItemString(dict, "mb_stride",    PyLong_FromLong(map->mb_stride));
    PyDict_SetItemString(dict, "sub_width",    PyLong_FromLong(map->sub_width));
    PyDict_SetItemString(dict, "sub_height",   PyLong_FromLong(map->sub_height));
    PyDict_SetItemString(dict, "sub_stride",   PyLong_FromLong(map->sub_stride));

    Py_DECREF(mb_arr);
    Py_DECREF(sub_arr);
    return dict;
}

static PyObject *
extract_hevc_bitcost(AVFrame *frame)
{
    if (!frame->opaque_ref || !frame->opaque_ref->data)
        Py_RETURN_NONE;

    if ((size_t)frame->opaque_ref->size < sizeof(HevcCtuBitCostMap))
        Py_RETURN_NONE;

    const HevcCtuBitCostMap *map = (const HevcCtuBitCostMap *)frame->opaque_ref->data;

    if (map->magic != HEVC_CTU_BIT_COST_MAP_MAGIC ||
        map->version != HEVC_CTU_BIT_COST_MAP_VERSION)
        Py_RETURN_NONE;

    size_t ctu_count = (size_t)map->ctb_stride * map->ctb_height;
    size_t sub_count = (size_t)map->sub_stride * map->sub_height;
    size_t expect_sz = sizeof(HevcCtuBitCostMap)
                        + ctu_count * sizeof(int32_t)
                        + sub_count * sizeof(float);

    if ((size_t)frame->opaque_ref->size < expect_sz)
        Py_RETURN_NONE;

    const int32_t *ctu_costs = hevc_ctu_costs_ptr(map);
    const float *sub_costs = hevc_sub_costs_ptr(map);

    npy_intp ctu_dims[2] = {map->ctb_height, map->ctb_stride};
    npy_intp sub_dims[2] = {map->sub_height, map->sub_stride};

    PyObject *ctu_arr = make_numpy_copy(2, ctu_dims, NPY_INT32,
                                        ctu_costs, ctu_count * sizeof(int32_t));
    PyObject *sub_arr = make_numpy_copy(2, sub_dims, NPY_FLOAT32,
                                        sub_costs, sub_count * sizeof(float));
    if (!ctu_arr || !sub_arr) {
        Py_XDECREF(ctu_arr);
        Py_XDECREF(sub_arr);
        return nullptr;
    }

    PyObject *dict = PyDict_New();
    if (!dict) {
        Py_DECREF(ctu_arr);
        Py_DECREF(sub_arr);
        return nullptr;
    }

    PyDict_SetItemString(dict, "ctu_bit_cost", ctu_arr);
    PyDict_SetItemString(dict, "sub_mb_bit_cost", sub_arr);
    PyDict_SetItemString(dict, "ctb_width", PyLong_FromLong(map->ctb_width));
    PyDict_SetItemString(dict, "ctb_height", PyLong_FromLong(map->ctb_height));
    PyDict_SetItemString(dict, "ctb_stride", PyLong_FromLong(map->ctb_stride));
    PyDict_SetItemString(dict, "log2_ctb_size", PyLong_FromLong(map->log2_ctb_size));
    PyDict_SetItemString(dict, "sub_width", PyLong_FromLong(map->sub_width));
    PyDict_SetItemString(dict, "sub_height", PyLong_FromLong(map->sub_height));
    PyDict_SetItemString(dict, "sub_stride", PyLong_FromLong(map->sub_stride));

    Py_DECREF(ctu_arr);
    Py_DECREF(sub_arr);
    return dict;
}

static PyObject *
extract_bitcost(AVFrame *frame)
{
    if (!frame->opaque_ref || !frame->opaque_ref->data)
        Py_RETURN_NONE;
    if ((size_t)frame->opaque_ref->size < sizeof(uint32_t))
        Py_RETURN_NONE;
    uint32_t magic = *(const uint32_t *)frame->opaque_ref->data;
    if (magic == H264_MB_BIT_COST_MAP_MAGIC)
        return extract_h264_bitcost(frame);
    if (magic == HEVC_CTU_BIT_COST_MAP_MAGIC)
        return extract_hevc_bitcost(frame);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Build one frame result dict                                         */
/* ------------------------------------------------------------------ */
static PyObject *
build_frame_dict(AVFrame *frame, int frame_count, AVCodecContext *dec_ctx,
                 int export_bitcost)
{
    PyObject *item = PyDict_New();
    if (!item)
        return nullptr;

    PyDict_SetItemString(item, "frame_idx", PyLong_FromLong(frame_count));

    const char *ptype = "?";
    if (frame->pict_type == AV_PICTURE_TYPE_I) ptype = "I";
    else if (frame->pict_type == AV_PICTURE_TYPE_P) ptype = "P";
    else if (frame->pict_type == AV_PICTURE_TYPE_B) ptype = "B";
    PyDict_SetItemString(item, "pict_type", PyUnicode_FromString(ptype));

    PyDict_SetItemString(item, "width",  PyLong_FromLong(frame->width));
    PyDict_SetItemString(item, "height", PyLong_FromLong(frame->height));

    const char *codec_name = dec_ctx->codec ? dec_ctx->codec->name : "unknown";
    PyDict_SetItemString(item, "codec_name", PyUnicode_FromString(codec_name));

    if (export_bitcost) {
        PyObject *bitcost = extract_bitcost(frame);
        if (!bitcost) {
            Py_DECREF(item);
            return nullptr;
        }
        PyDict_SetItemString(item, "bitcost", bitcost);
        Py_DECREF(bitcost);
    }

    return item;
}

/* ------------------------------------------------------------------ */
/* Python entry point                                                  */
/* ------------------------------------------------------------------ */
static PyObject *
read_video_fast(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *path = nullptr;
    int thread_count = 1;
    int export_bitcost = 0;
    const char *thread_type_str = "auto";

    static const char *kwlist[] = {"path", "thread_count", "export_bitcost", "thread_type", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|iis",
                                     (char **)kwlist,
                                     &path, &thread_count, &export_bitcost,
                                     &thread_type_str))
        return nullptr;

    int thread_type = FF_THREAD_FRAME;
    if (strcmp(thread_type_str, "slice") == 0) {
        thread_type = FF_THREAD_SLICE;
    } else if (strcmp(thread_type_str, "frame") == 0) {
        thread_type = FF_THREAD_FRAME;
    } else {
        // auto: slice when bitcost is needed (frame threading drops opaque_ref)
        thread_type = export_bitcost ? FF_THREAD_SLICE : FF_THREAD_FRAME;
    }

    AVFormatContext *fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path, nullptr, nullptr) < 0) {
        PyErr_SetString(PyExc_IOError, "Failed to open video file");
        return nullptr;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "Failed to find stream info");
        return nullptr;
    }

    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_idx < 0) {
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "No video stream found");
        return nullptr;
    }

    AVStream *st = fmt_ctx->streams[stream_idx];
    const AVCodec *dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "No H264/HEVC decoder found");
        return nullptr;
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(dec_ctx, st->codecpar);

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "skip_loop_filter", "all", 0);
    dec_ctx->thread_count = thread_count;
    dec_ctx->thread_type = thread_type;
    dec_ctx->skip_loop_filter = AVDISCARD_ALL;
    dec_ctx->skip_idct = AVDISCARD_ALL;

    if (avcodec_open2(dec_ctx, dec, &opts) < 0) {
        av_dict_free(&opts);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "Failed to open codec");
        return nullptr;
    }
    av_dict_free(&opts);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    PyObject *results = PyList_New(0);
    int frame_count = 0;

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            int ret = avcodec_send_packet(dec_ctx, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;

                PyObject *item = build_frame_dict(frame, frame_count, dec_ctx,
                                                    export_bitcost);
                if (!item) {
                    Py_DECREF(results);
                    results = nullptr;
                    goto cleanup;
                }
                PyList_Append(results, item);
                Py_DECREF(item);

                frame_count++;
                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    /* flush */
    avcodec_send_packet(dec_ctx, NULL);
    while (1) {
        int ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;

        PyObject *item = build_frame_dict(frame, frame_count, dec_ctx,
                                            export_bitcost);
        if (!item) {
            Py_DECREF(results);
            results = nullptr;
            goto cleanup;
        }
        PyList_Append(results, item);
        Py_DECREF(item);

        frame_count++;
        av_frame_unref(frame);
    }

cleanup:
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    return results;
}

/* ------------------------------------------------------------------ */
/* Python entry point: selected frames only                            */
/* ------------------------------------------------------------------ */
static PyObject *
read_video_fast_selected(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *path = nullptr;
    PyObject *frame_ids_obj = nullptr;
    int thread_count = 1;
    int export_bitcost = 0;
    const char *thread_type_str = "auto";

    static const char *kwlist[] = {"path", "frame_ids", "thread_count", "export_bitcost", "thread_type", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|iis",
                                     (char **)kwlist,
                                     &path, &frame_ids_obj, &thread_count, &export_bitcost,
                                     &thread_type_str))
        return nullptr;

    std::unordered_map<int, int> wanted_counts;
    int max_wanted = -1;
    PyObject *seq = PySequence_Fast(frame_ids_obj, "frame_ids must be a sequence of ints");
    if (!seq)
        return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *obj = PySequence_Fast_GET_ITEM(seq, i);
        long v = PyLong_AsLong(obj);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return nullptr;
        }
        if (v < 0)
            continue;
        wanted_counts[(int)v] += 1;
        if ((int)v > max_wanted)
            max_wanted = (int)v;
    }
    Py_DECREF(seq);

    PyObject *results = PyList_New(0);
    if (!results)
        return nullptr;
    if (wanted_counts.empty())
        return results;

    int thread_type = FF_THREAD_FRAME;
    if (strcmp(thread_type_str, "slice") == 0) {
        thread_type = FF_THREAD_SLICE;
    } else if (strcmp(thread_type_str, "frame") == 0) {
        thread_type = FF_THREAD_FRAME;
    } else {
        thread_type = export_bitcost ? FF_THREAD_SLICE : FF_THREAD_FRAME;
    }

    AVFormatContext *fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path, nullptr, nullptr) < 0) {
        Py_DECREF(results);
        PyErr_SetString(PyExc_IOError, "Failed to open video file");
        return nullptr;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        Py_DECREF(results);
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "Failed to find stream info");
        return nullptr;
    }

    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_idx < 0) {
        Py_DECREF(results);
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "No video stream found");
        return nullptr;
    }

    AVStream *st = fmt_ctx->streams[stream_idx];
    const AVCodec *dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
        Py_DECREF(results);
        avformat_close_input(&fmt_ctx);
        PyErr_SetString(PyExc_IOError, "No H264/HEVC decoder found");
        return nullptr;
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(dec_ctx, st->codecpar);

    std::vector<uint8_t> wanted_bitmap;
    if (max_wanted >= 0) {
        wanted_bitmap.assign((size_t)max_wanted + 2, 0);
        for (const auto &kv : wanted_counts) {
            if (kv.first >= 0 && kv.first <= max_wanted)
                wanted_bitmap[(size_t)kv.first] = 1;
        }
    }

    const char *disable_target_only_env = getenv("CVR_DISABLE_TARGET_ONLY");
    int cvr_disable_target_only = export_bitcost && thread_type == FF_THREAD_FRAME;
    if (disable_target_only_env && disable_target_only_env[0])
        cvr_disable_target_only = disable_target_only_env[0] != '0';

    CvrTargetBitcostCtx target_ctx;
    memset(&target_ctx, 0, sizeof(target_ctx));
    target_ctx.magic = CVR_TARGET_BITCOST_MAGIC;
    target_ctx.enabled = (export_bitcost && !cvr_disable_target_only) ? 1 : 0;
    target_ctx.max_frame = max_wanted;
    target_ctx.frame_bitmap = wanted_bitmap.empty() ? nullptr : wanted_bitmap.data();
    target_ctx.tolerance = 1;
    dec_ctx->opaque = &target_ctx;

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "skip_loop_filter", "all", 0);
    dec_ctx->thread_count = thread_count;
    dec_ctx->thread_type = thread_type;
    dec_ctx->skip_loop_filter = AVDISCARD_ALL;
    dec_ctx->skip_idct = AVDISCARD_ALL;

    if (avcodec_open2(dec_ctx, dec, &opts) < 0) {
        av_dict_free(&opts);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        Py_DECREF(results);
        PyErr_SetString(PyExc_IOError, "Failed to open codec");
        return nullptr;
    }
    av_dict_free(&opts);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    int frame_count = 0;
    int diag_selected_total = 0;
    int diag_selected_valid = 0;
    int diag_selected_none = 0;
    int diag_selected_zero = 0;
    bool stop = false;

    while (!stop && av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            int ret = avcodec_send_packet(dec_ctx, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0) {
                    Py_DECREF(results);
                    results = nullptr;
                    goto cleanup_selected;
                }

                auto it = wanted_counts.find(frame_count);
                if (it != wanted_counts.end() && it->second > 0) {
                    if (export_bitcost && cvr_bitcost_diag_enabled_cpp()) {
                        diag_selected_total++;
                        if (!cvr_bitcost_map_is_valid(frame))
                            diag_selected_none++;
                        else if (!cvr_bitcost_map_has_nonzero_mb(frame))
                            diag_selected_zero++;
                        else
                            diag_selected_valid++;
                    }
                    PyObject *item = build_frame_dict(frame, frame_count, dec_ctx,
                                                        export_bitcost);
                    if (!item) {
                        Py_DECREF(results);
                        results = nullptr;
                        goto cleanup_selected;
                    }
                    for (int k = 0; k < it->second; ++k) {
                        PyList_Append(results, item);
                    }
                    Py_DECREF(item);
                    wanted_counts.erase(it);
                    if (wanted_counts.empty() || frame_count >= max_wanted)
                        stop = true;
                }

                frame_count++;
                av_frame_unref(frame);
                if (stop)
                    break;
            }
        }
        av_packet_unref(pkt);
    }

    if (!stop) {
        avcodec_send_packet(dec_ctx, NULL);
        while (1) {
            int ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            auto it = wanted_counts.find(frame_count);
            if (it != wanted_counts.end() && it->second > 0) {
                PyObject *item = build_frame_dict(frame, frame_count, dec_ctx,
                                                    export_bitcost);
                if (!item) {
                    Py_DECREF(results);
                    results = nullptr;
                    goto cleanup_selected;
                }
                for (int k = 0; k < it->second; ++k) {
                    PyList_Append(results, item);
                }
                Py_DECREF(item);
                wanted_counts.erase(it);
                if (wanted_counts.empty() || frame_count >= max_wanted) {
                    av_frame_unref(frame);
                    break;
                }
            }
            frame_count++;
            av_frame_unref(frame);
        }
    }

cleanup_selected:
    if (export_bitcost && cvr_bitcost_diag_enabled_cpp()) {
        fprintf(stderr, "CVR_BITCOST_DIAG selected_total=%d valid=%d none=%d zero=%d decoded_until=%d remaining=%zu\n",
                diag_selected_total, diag_selected_valid, diag_selected_none, diag_selected_zero,
                frame_count, wanted_counts.size());
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    return results;
}

static PyMethodDef FastMethods[] = {
    {"read_video_fast", (PyCFunction)read_video_fast, METH_VARARGS | METH_KEYWORDS,
     "read_video_fast(path, thread_count=1, export_bitcost=0, thread_type='auto') -> list of frame metadata dicts"},
    {"read_video_fast_selected", (PyCFunction)read_video_fast_selected, METH_VARARGS | METH_KEYWORDS,
     "read_video_fast_selected(path, frame_ids, thread_count=1, export_bitcost=0, thread_type='auto') -> selected frame metadata dicts"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastmodule = {
    PyModuleDef_HEAD_INIT,
    "cv_reader_fast",
    "Fast stripped-down video decoder with optional bitcost export",
    -1,
    FastMethods
};

PyMODINIT_FUNC PyInit_cv_reader_fast(void) {
    PyObject *m = PyModule_Create(&fastmodule);
    import_array();   /* numpy – safe in C++ with numpy >= 1.19 */
    return m;
}
