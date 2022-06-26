#pragma once
#include "coasm.h"
#include <functional>

class OpuWarpinst;

namespace gem5 {
class OpuSimBase;

class OpuMemfetch {
 public:
  uint32_t size() { return data_size;};
  uint64_t get_pc() { return pc; };
  uint32_t data_size;
  uint64_t pc;
};

using icacheFetch_ftype = std::function<void(uint64_t, OpuMemfetch*)>;
using getLocalBaseVaddr_ftype = std::function<uint64_t()>;
using record_block_commit_ftype = std::function<void(uint32_t)>;
using executeMemOp_ftype = std::function<bool(const OpuWarpinst &)>;
using writebackClear_ftype = std::function<void()>;

#define CB(name) \
  virtual void setup_cb_##name(name##_ftype f) { \
      m_gem5_##name = f;                         \
  }                                              \
  name##_ftype m_gem5_##name;

class OpuCoreBase {
 public:
  // explicit OpuCoreBase(class OpuSimBase *gpu);
  virtual void cycle() = 0;
  virtual void setup_cb_icacheFetch(icacheFetch_ftype f) {
      m_gem5_icacheFetch = f;
  }
  virtual void setup_cb_getLocalBaseVaddr(getLocalBaseVaddr_ftype f) {
      m_gem5_getLocalBaseVaddr = f;
  }
  CB(record_block_commit)
  CB(executeMemOp)
  CB(writebackClear)

  virtual void accept_fetch_response(OpuMemfetch *mf) = 0;
  virtual bool ldst_unit_wb_inst(OpuWarpinst &inst) = 0;

  virtual void writeRegister(const OpuWarpinst &inst, unsigned warpSize, unsigned lane_id, char* data) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;

  virtual void warp_reaches_barrier(OpuWarpinst &inst) = 0;

  virtual bool fence_unblock_needed(unsigned warp_id) = 0;
  virtual void complete_fence(unsigned warp_id) = 0;

  virtual void finish_kernel() = 0;

  icacheFetch_ftype m_gem5_icacheFetch;
  getLocalBaseVaddr_ftype m_gem5_getLocalBaseVaddr;
};


}
