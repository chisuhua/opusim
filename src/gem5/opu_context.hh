#pragma once

namespace gem5 {
class OpuTop;
class OpuSimBase;
class OpuStream;
class OpuSimConfig;

class OpuContext {
 public:
  OpuContext() {
  }

  OpuSimBase *gem5_opu_sim_init(OpuStream **p_opu_stream, gem5::OpuTop *cuda_gpu, const char *config_path);
  OpuSimBase *g_the_opu {nullptr};
  OpuStream *g_opu_stream;
  OpuSimConfig *g_config;
};

}
