#pragma once
#include "funit.h"
class simtcore_config;

class sp_unit : public pipelined_simd_unit {
 public:
  sp_unit(register_set *result_port, const simtcore_config *config,
          shader_core_ctx *core, unsigned issue_reg_id, bool sub_core_model = true);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
        case opu_op_t::SFU_OP:
        return false;
        case opu_op_t::LOAD_OP:
        return false;
        case opu_op_t::TENSOR_LOAD_OP:
        return false;
        case opu_op_t::STORE_OP:
        return false;
        case opu_op_t::TENSOR_STORE_OP:
        return false;
        case opu_op_t::MEMORY_BARRIER_OP:
        return false;
        case opu_op_t::DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

