#pragma once
#include "simt_common.h"
#include "opuconfig.h"

namespace opu {

struct simt_core_stats_pod {
  void *simt_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  unsigned long long *shader_cycles;
  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  unsigned *m_last_num_sim_insn;
  unsigned *m_last_num_sim_winsn;
  unsigned * m_num_decoded_insn;  // number of instructions decoded by this shader core
  float *m_pipeline_duty_cycle;
  unsigned *m_num_FPdecoded_insn;
  unsigned *m_num_INTdecoded_insn;
  unsigned *m_num_storequeued_insn;
  unsigned *m_num_loadqueued_insn;
  unsigned *m_num_tex_inst;
  double *m_num_ialu_acesses;
  double *m_num_fp_acesses;
  double *m_num_imul_acesses;
  double *m_num_fpmul_acesses;
  double *m_num_idiv_acesses;
  double *m_num_fpdiv_acesses;
  double *m_num_sp_acesses;
  double *m_num_sfu_acesses;
  double *m_num_tensor_core_acesses;
  double *m_num_tex_acesses;
  double *m_num_const_acesses;
  double *m_num_dp_acesses;
  double *m_num_dpmul_acesses;
  double *m_num_dpdiv_acesses;
  double *m_num_sqrt_acesses;
  double *m_num_log_acesses;
  double *m_num_sin_acesses;
  double *m_num_exp_acesses;
  double *m_num_mem_acesses;
  unsigned *m_num_sp_committed;
  unsigned *m_num_tlb_hits;
  unsigned *m_num_tlb_accesses;
  unsigned *m_num_sfu_committed;
  unsigned *m_num_tensor_core_committed;
  unsigned *m_num_mem_committed;
  unsigned *m_read_regfile_acesses;
  unsigned *m_write_regfile_acesses;
  unsigned *m_non_rf_operands;
  double *m_num_imul24_acesses;
  double *m_num_imul32_acesses;
  unsigned *m_active_sp_lanes;
  unsigned *m_active_sfu_lanes;
  unsigned *m_active_tensor_core_lanes;
  unsigned *m_active_fu_lanes;
  unsigned *m_active_fu_mem_lanes;
  double *m_active_exu_threads; //For power model
  double *m_active_exu_warps; //For power model
  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  unsigned opu_n_load_insn;
  unsigned opu_n_store_insn;
  unsigned opu_n_shmem_insn;
  unsigned opu_n_sstarr_insn;
  unsigned opu_n_tex_insn;
  unsigned opu_n_const_insn;
  unsigned opu_n_param_insn;
  unsigned opu_n_shmem_bkconflict;
  unsigned opu_n_cache_bkconflict;
  int opu_n_intrawarp_mshr_merge;
  unsigned opu_n_cmem_portconflict;

  // unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
  //                                   [N_MEM_STAGE_STALL_TYPE];
  unsigned gpu_reg_bank_conflict_stalls;
  unsigned *shader_cycle_distro;
  unsigned *last_shader_cycle_distro;
  unsigned *num_warps_issuable;
  unsigned opu_n_stall_shd_mem;
  unsigned *single_issue_nums;
  unsigned *dual_issue_nums;

  unsigned ctas_completed;
  // memory access classification
  int opu_n_mem_read_local;
  int opu_n_mem_write_local;
  int opu_n_mem_texture;
  int opu_n_mem_const;
  int opu_n_mem_read_global;
  int opu_n_mem_write_global;
  int opu_n_mem_read_inst;

  int opu_n_mem_l2_writeback;
  int opu_n_mem_l1_write_allocate;
  int opu_n_mem_l2_write_allocate;

  unsigned made_write_mfs;
  unsigned made_read_mfs;

  unsigned *opu_n_shmem_bank_access;
  long *n_simt_to_mem;  // Interconnect power stats
  long *n_mem_to_simt;
};

class simt_core_stats : public simt_core_stats_pod {
 public:
  simt_core_stats(const simtcore_config *config) {
    m_config = config;
    simt_core_stats_pod *pod = reinterpret_cast<simt_core_stats_pod *>(
        this->simt_core_stats_pod_start);
    memset(pod, 0, sizeof(simt_core_stats_pod));
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long));
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float));
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tex_inst = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_ialu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_idiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpmul_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpdiv_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sfu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tensor_core_acesses = 
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tex_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sqrt_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_log_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sin_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_exp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_mem_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_hits = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_exu_warps =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_n_diverge = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    last_shader_cycle_distro =
        (unsigned *)calloc(m_config->warp_size + 3, sizeof(unsigned));
    single_issue_nums =
        (unsigned *)calloc(config->opu_num_sched_per_core, sizeof(unsigned));
    dual_issue_nums =
        (unsigned *)calloc(config->opu_num_sched_per_core, sizeof(unsigned));

    ctas_completed = 0;
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

    opu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader());
    m_shader_warp_slot_issue_distro.resize(config->num_shader());
  }

  ~simt_core_stats() {
    free(m_num_sim_insn);
    free(m_num_sim_winsn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
  }

  void new_grid() {}

  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  // void print(FILE *fout) const;

  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro;
  }

  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro;
  }

 private:
  const simtcore_config *m_config;

  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

  friend class power_stat_t;
  friend class simt_core_ctx;
  friend class ldst_unit;
  friend class opucore_cluster;
  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
};

}
