#pragma once
#include "warp_inst.h"
#include <string>

namespace opu {

class warp_inst_t;
class simt_core_ctx;

class simd_function_unit {
public:
  simd_function_unit( /*const simtcore_config *config */);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  virtual void issue( register_set& source_reg );
  virtual void cycle() = 0;
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  virtual bool can_issue(const warp_inst_t &inst) const ;
  virtual bool is_issue_partitioned() = 0;
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
/*
  virtual void print( FILE *fp ) const {
    fprintf(fp,"%s dispatch= ", m_name.c_str() );
    m_dispatch_reg->print(fp);
  }
*/
  const char *get_name() { return m_name.c_str(); }

 protected:
  std::string m_name;
  // const simtcore_config *m_config;
  warp_inst_t *m_dispatch_reg;
  static const unsigned MAX_ALU_LATENCY = 512;
  std::bitset<MAX_ALU_LATENCY> occupied;
};

class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      /*const simtcore_config *config,*/ unsigned max_latency,
                      simt_core_ctx *core, unsigned issue_reg_id, bool sub_core_model);

  // modifiers
  virtual void cycle();
  virtual void issue(register_set &source_reg);
  virtual unsigned get_active_lanes_in_pipeline();

  virtual void active_lanes_in_pipeline() = 0;
/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/
  // accessors
  virtual bool stallable() const { return false; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  /*
  virtual void print(FILE *fp) const {
    simd_function_unit::print(fp);
    for (int s = m_pipeline_depth - 1; s >= 0; s--) {
      if (!m_pipeline_reg[s]->empty()) {
        fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
        m_pipeline_reg[s]->print(fp);
      }
    }
  }
  */
 protected:
  unsigned m_pipeline_depth;
  warp_inst_t **m_pipeline_reg;
  register_set *m_result_port;
  class simt_core_ctx *m_core;
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
  bool m_sub_core_model;
};

}
