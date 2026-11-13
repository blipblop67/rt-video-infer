#pragma once
#include <string>

// Pure string building for GStreamer pipelines (no GStreamer dependency, unit tested).

// Everything up to (and including) an element nvvidconv-compatible caps can follow, for a
// source spec: test | file:PATH | image:PATH | csi:N | usb:DEV | mjpeg:DEV[@WxH] | rtsp://URL |
// pipeline:GST. Returns "" for an unknown spec.
std::string source_head(const std::string& spec);

// Full pipeline: head + RGBA caps at the network input size (in NVMM or system memory) + appsink.
// max-buffers=1 drop=true: always infer on the newest frame, never build a backlog.
std::string pipeline_description(const std::string& head, bool nvmm, int width, int height,
                                 bool sync);
