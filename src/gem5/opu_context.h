#ifndef __opu_context_h__
#define __opu_context_h__

class IsaSim;

class OpuContext {
 public:
  OpuContext() {
    isa_sim = new IsaSim(this);
  }

  IsaSim *isa_sim;
  IsaSim *get_isasim();
  class gpgpu_sim *gem5_ptx_sim_init_perf(stream_manager **p_stream_manager, gem5::CudaGPU *cuda_gpu, const char *config_path);
};

#endif /* __gpgpu_context_h__ */
