#include "check.h"
#include "source_spec.h"

namespace {
bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }
bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
}  // namespace

TEST(every_source_ends_in_nvvidconv) {
  // rtinfer appends RGBA caps + appsink after the head; that only negotiates after nvvidconv.
  for (const char* spec : {"test", "file:a.mp4", "image:a.jpg", "csi:0", "usb:/dev/video0",
                           "mjpeg:/dev/video0", "rtsp://cam/stream"})
    CHECK(ends_with(source_head(spec), "! nvvidconv"));
}

TEST(unknown_spec_is_rejected) {
  CHECK_EQ(source_head(""), std::string());
  CHECK_EQ(source_head("webcam"), std::string());
  CHECK_EQ(source_head("File:a.mp4"), std::string());  // prefixes are case-sensitive
}

TEST(file_uses_hardware_decoder_and_quotes_path) {
  const std::string h = source_head("file:/data/my clip.mp4");
  CHECK(contains(h, "location=\"/data/my clip.mp4\""));
  CHECK(contains(h, "nvv4l2decoder"));
}

TEST(usb_pins_30fps_raw) {
  CHECK_EQ(source_head("usb:/dev/video2"),
           std::string("v4l2src device=/dev/video2 ! video/x-raw,framerate=30/1 ! nvvidconv"));
}

TEST(mjpeg_defaults_to_720p_hardware_decode) {
  const std::string h = source_head("mjpeg:/dev/video0");
  CHECK(contains(h, "device=/dev/video0 "));
  CHECK(contains(h, "image/jpeg,width=1280,height=720,framerate=30/1"));
  CHECK(contains(h, "nvv4l2decoder mjpeg=1"));
}

TEST(mjpeg_parses_explicit_size) {
  const std::string h = source_head("mjpeg:/dev/video1@1920x1080");
  CHECK(contains(h, "device=/dev/video1 "));
  CHECK(contains(h, "width=1920,height=1080"));
}

TEST(mjpeg_rejects_malformed_size) {
  CHECK_EQ(source_head("mjpeg:/dev/video0@1920"), std::string());
  CHECK_EQ(source_head("mjpeg:/dev/video0@x1080"), std::string());
  CHECK_EQ(source_head("mjpeg:/dev/video0@1920x"), std::string());
}

TEST(rtsp_keeps_full_url) {
  CHECK(contains(source_head("rtsp://10.0.0.5:554/live"), "location=\"rtsp://10.0.0.5:554/live\""));
}

TEST(custom_pipeline_passes_through) {
  CHECK_EQ(source_head("pipeline:videotestsrc ! nvvidconv"), std::string("videotestsrc ! nvvidconv"));
}

TEST(pipeline_description_nvmm_and_sync) {
  CHECK_EQ(pipeline_description("SRC", true, 224, 224, true),
           std::string("SRC ! video/x-raw(memory:NVMM),format=RGBA,width=224,height=224"
                       " ! appsink name=sink max-buffers=1 drop=true sync=true"));
}

TEST(pipeline_description_cpu_no_sync) {
  const std::string d = pipeline_description("SRC", false, 320, 240, false);
  CHECK(contains(d, "! video/x-raw,format=RGBA,width=320,height=240 !"));
  CHECK(!contains(d, "NVMM"));
  CHECK(ends_with(d, "sync=false"));
}
