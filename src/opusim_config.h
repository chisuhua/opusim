#include <stdint.h>

namespace gem5 {
class OpuContext;
}

class simtcore_config;

class opu_sim_config {
 public:
  // opu_sim_config(gem5::OpuContext *ctx);
  opu_sim_config();

  ~opu_sim_config() ;

  void reg_options(class OptionParser *opp);
  void init() ;
  unsigned get_core_freq() const { return core_freq; }
  unsigned num_shader() const;
  unsigned num_cluster() const;
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }

  simtcore_config *get_shader_config() {
      return m_shader_config;
  }

 private:
  void init_clock_domains(void);
  // backward pointer
  // gem5::OpuContext *opu_ctx;
  bool m_valid;
  simtcore_config *m_shader_config;
  double core_freq;
  double icnt_freq;
  double dram_freq;
  double l2_freq;
  double core_period;
  double icnt_period;
  double dram_period;
  double l2_period;

  // GPGPU-Sim timing model options
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;
  bool opu_flush_l1_cache;
  bool opu_flush_l2_cache;
  bool opu_deadlock_detect;
  char *opu_clock_domains;
  int opu_cflog_interval;

  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;


  // Device Limits
  uint64_t stack_size_limit;
  uint64_t heap_size_limit;
  uint64_t runtime_sync_depth_limit;
  uint64_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int opu_compute_capability_major;
  unsigned int opu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class opu_sim;
};

