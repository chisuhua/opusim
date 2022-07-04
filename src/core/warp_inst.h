#pragma once
#ifndef __WARP_INST_H__
#define __WARP_INST_H__

#include "opuinst_base.h"
#include "coasm.h"
#include <vector>
#include <cstring>
#include <assert.h>
#include <stdint.h>
#include <memory>
#include <list>

#include "simt_common.h"
#include "mem_common.h"

class Instruction;
class WarpState;

namespace opu {

class simtcore_config;


class warp_inst_t : public OpuWarpinst {
public:
  warp_inst_t();

  uint32_t get_uid() const {};
  uint32_t get_schd_id() const {};
  void issue(const active_mask_t &mask, unsigned warp_id,
                        unsigned long long cycle, int dynamic_warp_id,
                        int sch_id);
  void clear() { m_empty=true; }
  void clear_active(const active_mask_t &inactive) ;
  void set_not_active(unsigned lane_id) ;
  void set_active(const active_mask_t &active) ;
  void Execute(WarpState* warp_state, uint32_t lane);
  uint32_t warp_id() const { return m_warp_id;};

  uint32_t GetSize() const;
  std::shared_ptr<Instruction> m_instruction;

  uint32_t m_uid;
  active_mask_t m_warp_active_mask;
  active_mask_t m_warp_issued_mask;
  uint32_t m_warp_id;
  uint32_t m_dynamic_warp_id;
  uint64_t issue_cycle;
  uint64_t cycles;
  bool m_empty;
  bool m_isatomic;
  bool m_cache_hit;
  bool should_do_atomic;
  uint32_t m_scheduler_id;  //the scheduler that issues this inst

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }

  void set_addr(unsigned n, address_type addr) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(MAX_WARP_SIZE);
      m_per_scalar_thread_valid = true;
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;
  }

  void set_addr(unsigned n, address_type *addr, unsigned num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(MAX_WARP_SIZE);
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
    for (unsigned i = 0; i < num_addrs; i++)
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];
  }

  // TODO schi add
  void set_data( unsigned n, const uint8_t *_data )
  {
    // assert( op == STORE_OP || memory_op == memory_store );
    // assert( space == global_space || space == const_space || space == local_space );
    assert( m_per_scalar_thread_valid );
    assert( !m_per_scalar_thread[n].data_valid );
    m_per_scalar_thread[n].data_valid = true;
    assert( _data );
    memcpy(&m_per_scalar_thread[n].data, _data, MAX_DATA_BYTES_PER_INSN_PER_THREAD);
  }

  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;
        data_valid = false;
    }
    address_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                       // upto 8 different
                                                       // requests (to support
                                                       // 32B access in 8 chunks
                                                       // of 4B each)
    // TODO schi add
    bool data_valid;
    uint8_t data[MAX_DATA_BYTES_PER_INSN_PER_THREAD];
  };
  bool m_per_scalar_thread_valid;
  std::vector<per_thread_info> m_per_scalar_thread;

  barrier_t bar_type;
  reduction_t red_type;
  uint32_t bar_id;
  uint32_t bar_count;
  uint32_t latency; // FIXME
  uint32_t initiation_interval {1};
  bool dispatch_delay() {
    if (cycles > 0) cycles--;
    return cycles > 0;
  }
  bool has_dispatch_delay() { return cycles > 0; }

  bool accessq_empty() const { return m_accessq.empty(); }
  unsigned accessq_count() const { return m_accessq.size(); }
  const mem_access_t &accessq_back() { return m_accessq.back(); }
  void accessq_pop_back() { m_accessq.pop_back(); }
  bool m_mem_accesses_created;
  std::list<mem_access_t> m_accessq;

  // special_ops sp_op;  // code (uarch visible) identify if int_alu, fp_alu, int_mul ....

  uint32_t get_num_operands() const ;
  uint32_t get_num_regs() const ;
  // return out reg idx
  int32_t out(uint32_t t) const;
  int32_t in(uint32_t t) const;

  int32_t incount() const;
  int32_t outcount() const;

  void generate_mem_accesses();
  void memory_coalescing_arch(bool is_write,
                                         mem_access_type access_type) ;

  opu_pipeline_t op_pipe;
  simtcore_config *m_config;
  void completed(unsigned long long cycle) const;
  void print(FILE *fout) const;
  void broadcast_barrier_reduction(const active_mask_t &access_mask);
};

void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

//register that can hold multiple instructions.
class register_set {
public:
  register_set(unsigned num, const char *name) {
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());
    }
    m_name = name;
  }
  const char *get_name() { return m_name; }
  bool has_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) return has_free();

    assert(reg_id < regs.size());
    return regs[reg_id]->empty();
  }
  bool has_ready() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model) return has_ready();
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());
  }

  unsigned get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    assert(has_ready());
    warp_inst_t **ready;
    ready = NULL;
    unsigned reg_id;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
          reg_id = i;
        }
      }
    }
    return reg_id;
  }
  unsigned get_schd_id(unsigned reg_id) {
    assert(not regs[reg_id]->empty());
    return regs[reg_id]->get_schd_id();
  }
  void move_in(warp_inst_t *&src) {
    warp_inst_t **free = get_free();
    move_warp(*free, src);
  }
  // void copy_in( warp_inst_t* src ){
  //   src->copy_contents_to(*get_free());
  //}
  void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);
    }
    move_warp(*free, src);
  }

  void move_out_to(warp_inst_t *&dest) {
    warp_inst_t **ready = get_ready();
    move_warp(dest, *ready);
  }
  void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);
    assert(ready != NULL);
    move_warp(dest, *ready);
  }

  warp_inst_t **get_ready() {
    warp_inst_t **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
        }
      }
    }
    return ready;
  }
  warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model) return get_ready();
    warp_inst_t **ready;
    ready = NULL;
    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty()) ready = &regs[reg_id];
    return ready;
  }
#if 0
  void print(FILE *fp) const {
    fprintf(fp, "%s : @%p\n", m_name, this);
    for (unsigned i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);
      fprintf(fp, "\n");
    }
  }
#endif
  warp_inst_t **get_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return &regs[i];
      }
    }
    assert(0 && "No free registers found");
    return NULL;
  }

  warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) return get_free();

    assert(reg_id < regs.size());
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];
    }
    assert(0 && "No free register found");
    return NULL;
  }

  unsigned get_size() { return regs.size(); }

private:
  std::vector<warp_inst_t *> regs;
  const char *m_name;
};

}

#endif
