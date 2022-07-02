#pragma once
#include "funit.h"
#include "warp_inst.h"
class simtcore_config;

class sf_unit : public pipelined_simd_unit {
 public:
  sf_unit(register_set *result_port, const simtcore_config *config,
      shader_core_ctx *core, unsigned issue_reg_id, bool sub_core_model = true);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
        case opu_op_t::SFU_OP:
        break;
        case opu_op_t::ALU_SFU_OP:
        break;
        case opu_op_t::DP_OP:
        break;  // for compute <= 29 (i..e Fermi and GT200)
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

