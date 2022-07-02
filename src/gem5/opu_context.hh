#pragma once

class opu_sim_config;
namespace gem5 {
class OpuTop;
class OpuSimBase;
class OpuStream;

class OpuContext {
 public:
  OpuContext() {
  }

  OpuSimBase *gem5_opu_sim_init(OpuStream **p_opu_stream, gem5::OpuTop *cuda_gpu, const char *config_path);
  OpuStream *m_opu_stream;
};

}
