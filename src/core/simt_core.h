#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

//#include "../cuda-sim/ptx.tab.h"

#include "abstract_core.h"
#include "delayqueue.h"
// #include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "tbsync.h"
#include "ifetch.h"
#include "opustats.h"
#include "opndcoll.h"
#include "opuconfig.h"
#include "l0_ccache.h"
// #include "stats.h"
class WarpState;
class WarpStateTest;

#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8
namespace gem5 {
class OpuContext;
}
class simt_core_stats;
class scheduler_unit;
class simd_function_unit;
class ldst_unit;
class inst_t;
class Scoreboard;
class opndcoll_rfu_t;
class KernelInfo;

enum exec_unit_type_t {
  NONE = 0,
  SP = 1,
  SFU = 2,
  MEM = 3,
  DP = 4,
  INT = 5,
  TENSOR = 6,
  SPECIALIZED = 7
};

class thread_ctx_t {
 public:
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;

  bool m_active;
};


inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;
};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;
};

class simt_core_ctx;
class simt_core_stats;

class simt_core_cluster;
class shader_memory_interface;
class simt_core_mem_fetch_allocator;
class cache_t;
namespace gem5 {
class OpuMemfetch;
}

#if 0
class simt_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  simt_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;
    m_cluster_id = cluster_id;
    m_memory_config = config;
  }
  mem_fetch *alloc(address_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle) const;
  mem_fetch *alloc(address_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf) const;
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy,
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    return mf;
  }

private:
  unsigned m_core_id;
  unsigned m_cluster_id;
  const memory_config *m_memory_config;
};
#endif

class simt_core_ctx : public core_t {
 public:
  // creator:
  simt_core_ctx(class opu_sim *gpu, class simt_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const simtcore_config *config,
                  simt_core_stats *stats);

  virtual ~simt_core_ctx() {};

  // used by simt_core_cluster:
  // modifiers
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  void issue_block2core(class KernelInfo &kernel);

  void cache_flush();
  void cache_invalidate();
  void accept_fetch_response(gem5::OpuMemfetch *mf);
  // void accept_ldst_unit_response(class mem_fetch *mf);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);

  // TODO schi add
  bool ldst_unit_wb_inst(OpuWarpinst &inst);

  void set_kernel(KernelInfo *k) ;

  // TODO schi 
  // Callback from gem5
  bool m_kernel_finishing;
  // void start_kernel_finish();
  void finish_kernel();
  bool kernel_finish_issued() { return m_kernel_finishing; }
  // PowerscalingCoefficients *scaling_coeffs;
  // accessors
  bool fetch_unit_response_buffer_full() const;
  bool ldst_unit_response_buffer_full() const;
  unsigned get_not_completed() const { return m_not_completed; }
  unsigned get_n_active_cta() const { return m_n_active_cta; }
  unsigned isactive() const {
    if (m_n_active_cta > 0)
      return 1;
    else
      return 0;
  }
  KernelInfo *get_kernel() { return m_kernel; }
  unsigned get_sid() const { return m_sid; }

  virtual warp_exec_t* get_warp(uint32_t warp_id);
  virtual WarpState* get_warp_state(uint32_t warp_id) const;

  // used by functional simulation:
  // modifiers
  // virtual void warp_exit(unsigned warp_id);
  virtual active_mask_t warp_active_mask(unsigned warp_id);

  // accessors
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;

  // TODO schi add
  void warp_reaches_barrier(OpuWarpinst &inst);
  bool fence_unblock_needed(unsigned warp_id) ;
  void complete_fence(unsigned warp_id) ;


  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  // used by pipeline timing model components:
  // modifiers
  void mem_instruction_stats(const warp_inst_t &inst);
  void decrement_atomic_count(unsigned wid, unsigned n);
  void inc_store_req(unsigned warp_id);
  void dec_inst_in_pipeline(unsigned warp_id) ;
  // void store_ack(class mem_fetch *mf);
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  void set_max_cta(const KernelInfo &kernel);
  void warp_inst_complete(const warp_inst_t &inst);

  // accessors
  std::list<unsigned> get_regs_written(const warp_inst_t &fvt) const;
  const simtcore_config *get_config() const { return m_config; }

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

  void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; }
  void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; }
  void incialu_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul24_stat(unsigned active_count,double latency) {
  if(m_config->opu_clock_gated_lanes==false){
    m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
   }
   void incimul32_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency);          
    }else{
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
   void incidiv_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
  }
   void incfpalu_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;     
  }
   void incfpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incfpdiv_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpalu_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++; 
   }
   void incdpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpdiv_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incsqrt_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void inclog_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incexp_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incsin_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }


   void inctensor_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctex_stat(unsigned active_count,double latency) {
    if(m_config->opu_clock_gated_lanes==false){
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inc_const_accesses(unsigned active_count) {
    m_stats->m_num_const_acesses[m_sid]=m_stats->m_num_const_acesses[m_sid]+active_count;
  }
#if 0
	 void inctrans_stat(unsigned active_count,double latency) {
		if(m_config->opu_clock_gated_lanes==false){
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency
			+ inactive_lanes_accesses_sfu(active_count, latency); 
		}else{
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency;
		}
	 }
#endif
  void incsfu_stat(unsigned active_count, double latency) {
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count*latency;
  }
  void incsp_stat(unsigned active_count, double latency) {
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count*latency;
  }
  void incmem_stat(unsigned active_count, double latency) {
    if (m_config->opu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency;
    }
  }
  // void incexecstat(warp_inst_t *&inst);

  void incregfile_reads(unsigned active_count) {
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) {
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) {
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

 protected:
  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) +
           (((32 - active_count) >> 3) * latency) +
           (((32 - active_count) >> 3) * latency);
  }
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency);
  }

  int test_res_bus(int latency);
  // address_type next_pc(int tid) const;
  void fetch();
  void register_cta_thread_exit(unsigned cta_num, KernelInfo *kernel);

  void decode();

  void issue();
  friend class scheduler_unit;  // this is needed to use private issue warp.
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
    friend class ldst_unit;
  void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

  void create_front_pipeline();
  void create_schedulers();
  void create_exec_pipeline();

  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          KernelInfo &kernel);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  virtual void func_exec_inst(warp_inst_t &inst) = 0;
/*
  virtual unsigned sim_init_thread(KernelInfo &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   opu_t *gpu) = 0;
*/
  virtual void create_shd_warp() = 0;

  virtual const warp_inst_t *get_next_inst(unsigned warp_id,
                                           address_type pc) = 0;
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;

  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   address_type *translated_addrs);

  void read_operands();

  void execute();

  void writeback();

  // used in display_pipeline():
  void dump_warp_state(FILE *fout) const;
  void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const simtcore_config *m_config;
  // const memory_config *m_memory_config;
  class simt_core_cluster *m_cluster;

  // statistics 
  simt_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  // thread_ctx_t *m_threadState;

  // interconnect interface
  // mem_fetch_interface *m_icnt;
  // simt_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  l0_ccache *m_L0C;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<std::shared_ptr<warp_exec_t>> m_warp;  // per warp information array
  std::vector<WarpState *> m_warp_state;  // per warp information array
  std::vector<WarpStateTest *> m_warp_statetest;  // per warp information array
  barrier_set_t m_barriers;
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // Jin: concurrent kernels on a sm
 public:
  bool can_issue_1block(KernelInfo &kernel);
  bool occupy_shader_resource_1block(KernelInfo &kernel, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, KernelInfo &kernel);
  int find_available_hwtid(unsigned int cta_size, bool occupy);

 private:
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};

class exec_simt_core_ctx : public simt_core_ctx {
 public:
  exec_simt_core_ctx(class opu_sim *gpu, class simt_core_cluster *cluster,
                       unsigned shader_id, unsigned tpc_id,
                       const simtcore_config *config,
                       simt_core_stats *stats)
      : simt_core_ctx(gpu, cluster, shader_id, tpc_id, config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void func_exec_inst(warp_inst_t &inst);
  /*
  virtual unsigned sim_init_thread(KernelInfo &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   opu_t *gpu);
                                   */
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
};

#if 0

class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(simt_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

 private:
  simt_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
 public:
  perfect_memory_interface(simt_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

 private:
  simt_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};
#endif
