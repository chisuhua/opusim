#pragma once
#include "coasm.h"
#include <functional>

namespace opu {
class warp_inst_t;
}

class OpuWarpinst;


namespace gem5 {
class OpuSimBase;

class OpuMemfetch {
 public:
  virtual uint32_t size() const = 0; // { return data_size;};
  virtual uint64_t get_pc() const = 0; // { return pc; };
  virtual uint8_t *get_data_ptr() = 0; // { return pc; };
  // uint32_t data_size;
  // uint64_t pc;
};

using icacheFetch_ftype = std::function<void(uint64_t, OpuMemfetch*)>;
using getLocalBaseVaddr_ftype = std::function<uint64_t()>;
using record_block_commit_ftype = std::function<void(uint32_t)>;
using executeMemOp_ftype = std::function<bool(const opu::warp_inst_t &)>;
using writebackClear_ftype = std::function<void()>;

#define CB(name) \
  void setup_cb_##name(name##_ftype f) { \
      m_gem5_##name = f;                         \
  }                                              \
  name##_ftype m_gem5_##name;

class OpuCoreBase {
 public:
  // explicit OpuCoreBase(class OpuSimBase *gpu);
  virtual ~OpuCoreBase() {};
  virtual void cycle() = 0;

  CB(icacheFetch)
  CB(getLocalBaseVaddr)
  CB(record_block_commit)
  CB(executeMemOp)
  CB(writebackClear)

  virtual void accept_ifetch_response(OpuMemfetch *mf) = 0;
  virtual bool ldst_unit_wb_inst(const OpuWarpinst &inst) = 0;
  virtual void writeRegister(const OpuWarpinst &inst, unsigned warpSize, unsigned lane_id, char* data) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  virtual void warp_reaches_barrier(const OpuWarpinst &inst) = 0;
  virtual bool fence_unblock_needed(unsigned warp_id) = 0;
  virtual void complete_fence(unsigned warp_id) = 0;

  virtual void finish_kernel() = 0;

};


}
