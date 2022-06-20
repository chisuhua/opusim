#pragma once
#include "coasm.h"

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


class OpuCoreBase {
 public:
  explicit OpuCoreBase(class OpuSimBase *gpu);
  virtual void cycle() = 0;
  virtual void accept_fetch_response(OpuMemfetch *mf) = 0;
  virtual bool ldst_unit_wb_inst(OpuWarpinst &inst) = 0;

  virtual void writeRegister(const OpuWarpinst &inst, unsigned warpSize, unsigned lane_id, char* data) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;

  virtual void warp_reaches_barrier(OpuWarpinst &inst) = 0;

  virtual bool fence_unblock_needed(unsigned warp_id) = 0;
  virtual void complete_fence(unsigned warp_id) = 0;

  virtual void finish_kernel() = 0;
};


}
