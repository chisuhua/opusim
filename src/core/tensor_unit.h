#pragma once
#include "funit.h"

namespace opu {
class simtcore_config;

class tensor_unit : public pipelined_simd_unit {
 public:
  tensor_unit(register_set *result_port, const simtcore_config *config,
              simt_core_ctx *core, unsigned issue_reg_id, bool sub_core_model = true);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
        case opu_op_t::TENSOR_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};
}
