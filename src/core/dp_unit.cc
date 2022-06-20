#include "dp_unit.h"
#include "simt_core.h"
#include "warp_inst.h"
#include "coasm.h"


dp_unit::dp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id, bool sub_core_model)
    : pipelined_simd_unit(result_port, config->max_dp_latency, core,
                          issue_reg_id, sub_core_model) {
  m_name = "DP ";
}

void dp_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  //m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void dp_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = opu_pipeline_t::DP__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}
