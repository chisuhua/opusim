#pragma once
#include "funit.h"
#include "stats.h"
#include <list>

class simtcore_config;
class Scoreboard;
class opndcoll_rfu_t;
class simt_core_stats;
class l0_ccache;
class l1_cache;

class ldst_unit: public pipelined_simd_unit {
public:
  ldst_unit(/*mem_fetch_interface *icnt,
            simt_core_mem_fetch_allocator *mf_allocator,*/
            simt_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, uint32_t smem_latency,
            simt_core_stats *stats,
            unsigned sid, unsigned tpc, bool sub_core_model = true);

  // modifiers
  virtual void issue(register_set &inst);
  bool is_issue_partitioned() { return false; }
  virtual void cycle();

  //void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback();

  // TODO schi add
  /// Inserts this instruction into the writeback stage of the pipeline
  /// Returns true if successful, false if there is an instruction blocking
  bool writebackInst(warp_inst_t &inst);

  // accessors
  virtual unsigned clock_multiplier() const;
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
        case opu_op_t::LOAD_OP:
        break;
        case opu_op_t::TENSOR_LOAD_OP:
        break;
        case opu_op_t::STORE_OP:
        break;
        case opu_op_t::TENSOR_STORE_OP:
        break;
        case opu_op_t::MEMORY_BARRIER_OP:
        break;
      default:
        return false;
    }
    return m_dispatch_reg->empty();
  }

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  // void print(FILE *fout) const;

protected:
#if 0
  ldst_unit(/*mem_fetch_interface *icnt,
            simt_core_mem_fetch_allocator *mf_allocator,*/
            simt_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard,
            /*simt_core_stats *stats,*/
            unsigned sid, unsigned tpc);
#endif
  void init(/*mem_fetch_interface *icnt,
            simt_core_mem_fetch_allocator *mf_allocator,*/
            simt_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard,
            simt_core_stats *stats,
            unsigned sid, unsigned tpc);


 protected:
  bool shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);
  // bool constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
  //                    mem_stage_access_type &fail_type);
  // bool texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
  //                   mem_stage_access_type &fail_type);
  // bool memory_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
  //                  mem_stage_access_type &fail_type);

   bool memory_cycle_gem5( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

#if 0
  virtual mem_stage_stall_type process_cache_access(
      cache_t *cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t *cache,
                                                   warp_inst_t &inst);
  mem_stage_stall_type process_memory_access_queue_l1cache(l1_cache *cache,
                                                           warp_inst_t &inst);
#endif
  simtcore_config *m_config;
  // const memory_config *m_memory_config;
  // class mem_fetch_interface *m_icnt;
  // simt_core_mem_fetch_allocator *m_mf_allocator;
  class simt_core_ctx *m_core;
  unsigned m_sid;
  unsigned m_tpc;

  l0_ccache *m_L0C; // constant cache
  l1_cache *m_L0S; // data cache
  l1_cache *m_L0V; // data cache

  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/>>
      m_pending_writes;
  std::list<mem_fetch*> m_response_fifo;
  opndcoll_rfu_t *m_operand_collector;
  Scoreboard *m_scoreboard;

  mem_fetch *m_next_global;
  warp_inst_t m_next_wb;
  unsigned m_writeback_arb;  // round-robin arbiter for writeback contention
                             // between L1T, L1C, shared
  unsigned m_num_writeback_clients;

  // enum mem_stage_stall_type m_mem_rc;

  simt_core_stats *m_stats;

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // std::vector<std::deque<mem_fetch *>> l1_latency_queue;
  // void L1_latency_queue_cycle();
};

