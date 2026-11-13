#include "source_spec.h"

namespace {
bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
}  // namespace

std::string source_head(const std::string& spec) {
  if (spec == "test")
    return "videotestsrc is-live=true pattern=ball ! "
           "video/x-raw,width=1280,height=720,framerate=30/1 ! nvvidconv";
  if (starts_with(spec, "file:"))  // H.264 MP4 decoded on NVDEC
    return "filesrc location=\"" + spec.substr(5) +
           "\" ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv";
  if (starts_with(spec, "image:"))  // still image replayed as a 30 fps live stream
    return "filesrc location=\"" + spec.substr(6) +
           "\" ! jpegdec ! videoconvert ! video/x-raw,format=I420 ! "
           "imagefreeze is-live=true ! video/x-raw,framerate=30/1 ! nvvidconv";
  if (starts_with(spec, "csi:"))
    return "nvarguscamerasrc sensor-id=" + spec.substr(4) +
           " ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv";
  if (starts_with(spec, "usb:"))  // raw YUY2; 30 fps pins UVC cameras to a mode that sustains it
    return "v4l2src device=" + spec.substr(4) + " ! video/x-raw,framerate=30/1 ! nvvidconv";
  if (starts_with(spec, "mjpeg:")) {  // mjpeg:/dev/video0[@WxH], decoded on the hardware JPEG engine
    std::string dev = spec.substr(6), w = "1280", h = "720";
    const size_t at = dev.rfind('@');
    if (at != std::string::npos) {
      const size_t x = dev.find('x', at);
      if (x == std::string::npos || x == at + 1 || x + 1 == dev.size()) return {};  // malformed size
      w = dev.substr(at + 1, x - at - 1);
      h = dev.substr(x + 1);
      dev = dev.substr(0, at);
    }
    return "v4l2src device=" + dev + " ! image/jpeg,width=" + w + ",height=" + h +
           ",framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv";
  }
  if (starts_with(spec, "rtsp:"))
    return "rtspsrc location=\"" + spec + "\" latency=100 ! rtph264depay ! h264parse ! "
           "nvv4l2decoder ! nvvidconv";
  if (starts_with(spec, "pipeline:")) return spec.substr(9);
  return {};
}

std::string pipeline_description(const std::string& head, bool nvmm, int width, int height,
                                 bool sync) {
  return head + " ! video/x-raw" + (nvmm ? "(memory:NVMM)" : "") +
         ",format=RGBA,width=" + std::to_string(width) + ",height=" + std::to_string(height) +
         " ! appsink name=sink max-buffers=1 drop=true sync=" + (sync ? "true" : "false");
}
