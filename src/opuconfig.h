#pragma once
#include "simt_common.h"
#include <algorithm>
#include <sstream>
#include <cstring>
#include <cassert>

namespace gem5 {
  class OpuContext;
}

class core_config {
 public:
  core_config() {
    m_valid = false;
    num_shmem_bank = 16;
    shmem_limited_broadcast = false;
    opu_shmem_sizeDefault = (unsigned)-1;
    opu_shmem_sizePrefL1 = (unsigned)-1;
    opu_shmem_sizePrefShared = (unsigned)-1;
  }
  virtual void init() = 0;

  bool m_valid;
  unsigned warp_size;
  // off-chip memory request architecture parameters
  int opu_coalesce_arch;

  // shared memory bank conflict checking parameters
  bool shmem_limited_broadcast;
  static const address_type WORD_SIZE = 4;
  unsigned num_shmem_bank;
  unsigned shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);
  }
  unsigned mem_warp_parts;
  mutable unsigned opu_shmem_size;
  char *opu_shmem_option;
  std::vector<unsigned> shmem_opt_list;
  unsigned opu_shmem_sizeDefault;
  unsigned opu_shmem_sizePrefL1;
  unsigned opu_shmem_sizePrefShared;
  unsigned mem_unit_ports;

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  unsigned opu_cache_texl1_linesize;
  unsigned opu_cache_constl1_linesize;

  unsigned opu_max_insn_issue_per_warp;
  bool gmem_skip_L1D;  // on = global memory access always skip the L1 cache

  bool adaptive_cache_config;
};

class shader_core_config : public core_config {
 public:
  shader_core_config() : core_config() {
    pipeline_widths_string = NULL;
  }

  static shader_core_config* getInstance() {
    static shader_core_config* config = nullptr;
    if (config == nullptr) {
      config = new shader_core_config();
      config->init();
    }
    return config;
  }

  void init() {
    int ntok = sscanf(opu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size);
    if (ntok != 2) {
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "opu_shader_core_pipeline_opt\n");
      abort();
    }

    char *toks = new char[100];
    char *tokd = toks;
    strcpy(toks, pipeline_widths_string);

    toks = strtok(toks, ",");

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!opu_tensor_core_avail);

    for (int i = 0; i < num_config_to_read; i++) {
      assert(toks);
      ntok = sscanf(toks, "%d", &pipe_widths[i]);
      assert(ntok == 1);
      toks = strtok(NULL, ",");
    }

    delete[] tokd;

    if (n_thread_per_shader > MAX_THREAD_PER_SM) {
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_core.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort();
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam;
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name);

      if (enabled) {
        m_specialized_unit.push_back(sparam);
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name));
        m_specialized_unit_num += sparam.num_units;
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
    }

    // parse opu_shmem_option for adpative cache config
    if (adaptive_cache_config) {
      std::stringstream ss(opu_shmem_option);
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ',');
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024);
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end());
    }
  }
  void reg_options(class OptionParser *opp);
  // unsigned max_cta(const kernel_info_t &k) const;
  unsigned num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster;
  }
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid;
  }
  void set_pipeline_latency();

  // backward pointer
  class OpuContext *opu_ctx;
  // data
  char *opu_shader_core_pipeline_opt;
  bool opu_perfect_mem;
  bool opu_clock_gated_reg_file;
  bool opu_clock_gated_lanes;
  // enum divergence_support_t model;
  unsigned n_thread_per_shader;
  unsigned n_regfile_gating_group;
  unsigned max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *opu_scheduler_string;
  unsigned opu_shmem_per_block;
  unsigned opu_registers_per_block;
  char *pipeline_widths_string;
  int pipe_widths[N_PIPELINE_STAGES];

  bool opu_dwf_reg_bankconflict;

  unsigned opu_num_sched_per_core;
  int opu_max_insn_issue_per_warp;
  bool opu_dual_issue_diff_exec_units;

  // op collector
  bool enable_specialized_operand_collector;
  int opu_operand_collector_num_units_sp;
  int opu_operand_collector_num_units_dp;
  int opu_operand_collector_num_units_sfu;
  int opu_operand_collector_num_units_tensor_core;
  int opu_operand_collector_num_units_mem;
  int opu_operand_collector_num_units_gen;
  int opu_operand_collector_num_units_int;

  unsigned int opu_operand_collector_num_in_ports_sp;
  unsigned int opu_operand_collector_num_in_ports_dp;
  unsigned int opu_operand_collector_num_in_ports_sfu;
  unsigned int opu_operand_collector_num_in_ports_tensor_core;
  unsigned int opu_operand_collector_num_in_ports_mem;
  unsigned int opu_operand_collector_num_in_ports_gen;
  unsigned int opu_operand_collector_num_in_ports_int;

  unsigned int opu_operand_collector_num_out_ports_sp;
  unsigned int opu_operand_collector_num_out_ports_dp;
  unsigned int opu_operand_collector_num_out_ports_sfu;
  unsigned int opu_operand_collector_num_out_ports_tensor_core;
  unsigned int opu_operand_collector_num_out_ports_mem;
  unsigned int opu_operand_collector_num_out_ports_gen;
  unsigned int opu_operand_collector_num_out_ports_int;

  int opu_num_sp_units;
  int opu_tensor_core_avail;
  int opu_num_dp_units;
  int opu_num_sfu_units;
  int opu_num_tensor_core_units;
  int opu_num_mem_units;
  int opu_num_int_units;

  // Shader core resources
  unsigned opu_shader_registers;
  int opu_warpdistro_shader;
  int opu_warp_issue_shader;
  unsigned opu_num_reg_banks;
  bool opu_reg_bank_use_warp_id;
  bool opu_local_mem_map;
  bool opu_ignore_resources_limitation;
  bool sub_core_model;

  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;

  unsigned n_simt_cores_per_cluster;
  unsigned n_simt_clusters;
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  char *opcode_latency_int;
  char *opcode_latency_fp;
  char *opcode_latency_dp;
  char *opcode_latency_sfu;
  char *opcode_latency_tensor;
  char *opcode_initiation_int;
  char *opcode_initiation_fp;
  char *opcode_initiation_dp;
  char *opcode_initiation_sfu;
  char *opcode_initiation_tensor;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  bool opu_concurrent_kernel_sm;

  bool perfect_inst_const_cache;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  unsigned m_specialized_unit_num;
};

