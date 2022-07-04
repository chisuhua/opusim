#pragma once

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include "core/abstract_core.h"
//#include "../option_parser.h"
//#include "../trace.h"
//#include "addrdec.h"
//#include "gpu-cache.h"
#include "core/opucore.h"
#include "opuusim_base.h"
#include "inc/ExecContext.h"

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

// extern tr1_hash_map<address_type, unsigned> address_random_interleaving;


extern bool g_interactive_debugger_enabled;


#if 0
struct occupancy_stats {
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),
        aggregate_theoretical_warp_slots(tws) {}

  unsigned long long aggregate_warp_slot_filled;
  unsigned long long aggregate_theoretical_warp_slots;

  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;
    return *this;
  }

  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};
#endif

namespace gem5 {
class OpuContext;
class OpuTop;
class OpuCoreBase;
class KernelInfoBase;
}
class KernelInfo;


namespace opu {
class opu_sim_config;

class opu_sim : public ::gem5::OpuSimBase {
 public:
  opu_sim(opu_sim_config *config, ::gem5::OpuContext *ctx, ::gem5::OpuTop *opu_top = NULL );

  void set_prop(struct cudaDeviceProp *prop);

  void launch(::gem5::KernelInfoBase *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(KernelInfo *kernel);
  void stop_all_running_kernels();
  uint32_t num_cores();

  void init();
  // void cycle();
  void core_cycle_start();
  void core_cycle_end();
  // void icnt_cycle_start();
  // void icnt_cycle_end();
  // void l2_cycle();
  // void dram_cycle();

  bool active();
  bool cycle_insn_cta_max_hit() {
      return false;
      /*
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
            */
  }
  void print_stats();
  void update_stats();
  void deadlock_check();
  void inc_completed_cta() { gpu_completed_cta++; }
  // void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
  //                             unsigned *rpc);

  uint32_t shared_mem_size() const;
  int shared_mem_per_block() const;
  int compute_capability_major() const;
  int compute_capability_minor() const;
  uint32_t num_registers_per_core() const;
  int num_registers_per_block() const;
  int warp_size() const;
  uint32_t shader_clock() const;
  int max_cta_per_core() const;
  int get_max_cta(const KernelInfo &k) const;
  const struct cudaDeviceProp *get_prop() const;
  enum divergence_support_t simd_model() const;

  unsigned threads_per_core() const;
  bool get_more_cta_left() const;
  bool kernel_more_cta_left(KernelInfo *kernel) const;
  bool hit_max_cta_count() const;
  KernelInfo *select_kernel();
  // PowerscalingCoefficients *get_scaling_coeffs();
  void decrement_kernel_latency();

  const opu_sim_config *get_config() const { return &m_config; }
  void gpu_print_stat();
  void dump_pipeline(int mask, int s, int m) const;

  // TODO schi add
  ::gem5::OpuCoreBase* getSIMTCore(uint32_t id) override;

  // void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // The next three functions added to be used by the functional simulation
  // function

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const simtcore_config *getShaderCoreConfig();

  //! Get shader core SIMT cluster
  /*!
   * Returning the cluster of of the shader core, used by the functional
   * simulation so far
   */
  opucore_cluster *getSIMTCluster();

  // backward pointer
  ::gem5::OpuContext *opu_ctx;

 private:
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();
  void print_dram_stats(FILE *fout) const;
  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;
  void visualizer_printstat();
  void print_shader_cycle_distro(FILE *fout) const;

  void opu_debug();

 protected:
  ///// data /////
  class opucore_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::vector<KernelInfo *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;

  unsigned m_last_cluster_issue;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;

  // debug
  bool gpu_deadlock;

  //// configuration parameters ////
  opu_sim_config &m_config;

  const struct cudaDeviceProp *m_cuda_properties;
  const simtcore_config *m_shader_config;
  // const memory_config *m_memory_config;

  // stats
  class simt_core_stats *m_shader_stats;
  // class memory_stats_t *m_memory_stats;
  // class power_stat_t *m_power_stats;
  class opu_sim_wrapper *m_gpgpusim_wrapper;
  unsigned long long last_gpu_sim_insn;

  unsigned long long last_liveness_message_time;

  std::map<std::string, FuncCache> m_special_cache_config;

  std::vector<std::string> m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned> m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  // std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  // std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  // std::string executed_kernel_name();
  // void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  virtual void createSIMTCluster() = 0;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  // occupancy_stats gpu_occupancy;
  // occupancy_stats gpu_tot_occupancy;

  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;

  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);

  // Jin: functional simulation for CDP
 private:
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  KernelInfo *m_functional_sim_kernel;

 public:
  bool is_functional_sim() { return m_functional_sim; }
  KernelInfo *get_functional_kernel() { return m_functional_sim_kernel; }
  void functional_launch(KernelInfo *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(KernelInfo *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }
};

class exec_opu_sim : public opu_sim {
 public:
  exec_opu_sim(opu_sim_config *config, ::gem5::OpuContext *ctx, ::gem5::OpuTop *opu_top = NULL )
      : opu_sim(config, ctx, opu_top) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster() override;
};
}
