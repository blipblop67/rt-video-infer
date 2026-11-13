#include "gst_source.h"

#include <cstdio>

#include <cudaEGL.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>

#include "source_spec.h"


GstSource::~GstSource() { close(); }

GstPadProbeReturn GstSource::count_probe(GstPad*, GstPadProbeInfo*, gpointer self) {
  static_cast<GstSource*>(self)->arrived_++;
  return GST_PAD_PROBE_OK;
}

GstFlowReturn GstSource::on_new_sample(GstElement*, gpointer self) {
  static_cast<GstSource*>(self)->cfg_.on_ready();  // just a wake-up; the gatherer pulls
  return GST_FLOW_OK;
}

void GstSource::on_eos(GstElement*, gpointer self) { static_cast<GstSource*>(self)->cfg_.on_ready(); }

bool GstSource::open(const SourceConfig& cfg) {
  cfg_ = cfg;
  const std::string head = source_head(cfg.spec);
  if (head.empty()) {
    fprintf(stderr, "unknown source spec '%s'\n", cfg.spec.c_str());
    return false;
  }
  desc_ = pipeline_description(head, cfg.mem == MemMode::kNvmm, cfg.width, cfg.height, cfg.sync);

  GError* err = nullptr;
  pipeline_ = gst_parse_launch(desc_.c_str(), &err);
  if (!pipeline_ || err) {
    fprintf(stderr, "pipeline parse failed: %s\n", err ? err->message : "?");
    if (err) g_error_free(err);
    return false;
  }
  sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  bus_ = gst_element_get_bus(pipeline_);

  // Count every frame that reaches appsink, so frames dropped by appsink are visible.
  GstPad* pad = gst_element_get_static_pad(sink_, "sink");
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, count_probe, this, nullptr);
  gst_object_unref(pad);
  if (cfg_.on_ready) {
    g_object_set(sink_, "emit-signals", TRUE, nullptr);
    g_signal_connect(sink_, "new-sample", G_CALLBACK(on_new_sample), this);
    g_signal_connect(sink_, "eos", G_CALLBACK(on_eos), this);
  }

  if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    poll_bus();
    fprintf(stderr, "failed to start pipeline\n");
    return false;
  }
  return true;
}

void GstSource::close() {
  clear_egl_cache();
  if (pipeline_) {
    // Stopping mid-stream: send EOS and give it a moment to drain, so muxers in custom pipelines
    // (e.g. a tee'd recording branch) finalize their files instead of leaving them truncated.
    if (!eos_ && !error_ && gst_element_send_event(pipeline_, gst_event_new_eos())) {
      if (GstMessage* m = gst_bus_timed_pop_filtered(
              bus_, 3 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR)))
        gst_message_unref(m);
    }
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (sink_) gst_object_unref(sink_), sink_ = nullptr;
  if (bus_) gst_object_unref(bus_), bus_ = nullptr;
}

void GstSource::poll_bus() {
  while (GstMessage* msg = gst_bus_pop_filtered(
             bus_, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS))) {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
      eos_ = true;
    } else {
      GError* e = nullptr;
      gchar* dbg = nullptr;
      gst_message_parse_error(msg, &e, &dbg);
      fprintf(stderr, "GStreamer error from %s: %s\n  %s\n", GST_OBJECT_NAME(msg->src),
              e->message, dbg ? dbg : "");
      g_error_free(e);
      g_free(dbg);
      error_ = true;
    }
    gst_message_unref(msg);
  }
}

GstSample* GstSource::pull(int timeout_ms) {
  GstSample* s =
      gst_app_sink_try_pull_sample(GST_APP_SINK(sink_), (GstClockTime)timeout_ms * GST_MSECOND);
  if (!s) {
    poll_bus();
    if (gst_app_sink_is_eos(GST_APP_SINK(sink_))) eos_ = true;
  }
  return s;
}

bool GstSource::acquire(GstSample* sample, FrameView& f) {
  f = FrameView{};
  f.sample = sample;
  f.buffer = gst_sample_get_buffer(sample);
  if (!gst_buffer_map(f.buffer, &f.map, GST_MAP_READ)) return false;

  if (cfg_.mem == MemMode::kCpu) {
    GstVideoInfo info;
    gst_video_info_from_caps(&info, gst_sample_get_caps(sample));
    GstVideoMeta* meta = gst_buffer_get_video_meta(f.buffer);
    f.host = f.map.data + (meta ? meta->offset[0] : 0);
    f.pitch = meta ? meta->stride[0] : GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    return true;
  }

  // NVMM: the mapped data is an NvBufSurface; expose it to CUDA through an EGLImage.
  f.surf = reinterpret_cast<NvBufSurface*>(f.map.data);
  if (!cfg_.cache_egl) return register_surface(f.surf, f.egl_res, f.dev, f.pitch);

  const int fd = static_cast<int>(f.surf->surfaceList[0].bufferDesc);
  auto it = egl_cache_.find(f.surf);
  if (it != egl_cache_.end() && it->second.fd != fd) {  // pool was reallocated: drop stale entry
    cuGraphicsUnregisterResource(it->second.res);
    egl_cache_.erase(it);
    it = egl_cache_.end();
  }
  if (it == egl_cache_.end()) {
    EglEntry e{fd, nullptr, nullptr, 0};
    if (!register_surface(f.surf, e.res, e.dev, e.pitch)) return false;
    it = egl_cache_.emplace(f.surf, e).first;
  }
  f.egl_res = it->second.res;
  f.dev = it->second.dev;
  f.pitch = it->second.pitch;
  f.cached = true;
  return true;
}

bool GstSource::register_surface(NvBufSurface* surf, CUgraphicsResource& res,
                                 const uint8_t*& dev, int& pitch) {
  if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
    fprintf(stderr, "NvBufSurfaceMapEglImage failed\n");
    return false;
  }
  EGLImageKHR img = surf->surfaceList[0].mappedAddr.eglImage;
  if (cuGraphicsEGLRegisterImage(&res, img, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) !=
      CUDA_SUCCESS) {
    fprintf(stderr, "cuGraphicsEGLRegisterImage failed\n");
    return false;
  }
  CUeglFrame egl{};
  if (cuGraphicsResourceGetMappedEglFrame(&egl, res, 0, 0) != CUDA_SUCCESS ||
      egl.frameType != CU_EGL_FRAME_TYPE_PITCH) {
    fprintf(stderr, "EGL frame is not pitch-linear\n");
    return false;
  }
  dev = static_cast<const uint8_t*>(egl.frame.pPitch[0]);
  pitch = static_cast<int>(egl.pitch);
  return true;
}

// Must run while the pipeline still owns its buffer pool (before going to NULL).
void GstSource::clear_egl_cache() {
  for (auto& [surf, e] : egl_cache_) {
    cuGraphicsUnregisterResource(e.res);
    if (surf->surfaceList[0].mappedAddr.eglImage) NvBufSurfaceUnMapEglImage(surf, 0);
  }
  egl_cache_.clear();
}

// Caller must have synchronized any GPU work reading f.dev before releasing.
void GstSource::release(FrameView& f) {
  if (!f.cached) {
    if (f.egl_res) cuGraphicsUnregisterResource(f.egl_res);
    if (f.surf && f.surf->surfaceList[0].mappedAddr.eglImage) NvBufSurfaceUnMapEglImage(f.surf, 0);
  }
  if (f.buffer && f.map.data) gst_buffer_unmap(f.buffer, &f.map);
  if (f.sample) gst_sample_unref(f.sample);
  f = FrameView{};
}

double GstSource::latency_ms(GstSample* sample) const {
  GstBuffer* buf = gst_sample_get_buffer(sample);
  GstSegment* seg = gst_sample_get_segment(sample);
  GstClock* clock = gst_element_get_clock(pipeline_);
  if (!clock || !seg || !GST_BUFFER_PTS_IS_VALID(buf)) {
    if (clock) gst_object_unref(clock);
    return -1;
  }
  const GstClockTime now = gst_clock_get_time(clock) - gst_element_get_base_time(pipeline_);
  gst_object_unref(clock);
  const guint64 rt = gst_segment_to_running_time(seg, GST_FORMAT_TIME, GST_BUFFER_PTS(buf));
  if (rt == GST_CLOCK_TIME_NONE || now < rt) return -1;
  return (now - rt) / 1e6;
}
