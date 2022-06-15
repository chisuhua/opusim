#pragma once
#include "coasm.h"

namespace gem5 {
class OpuSimBase;

class OpuMemfetch {
 public:
  uint32_t size() { return data_size;};
  uint64_t get_pc() { return pc; };
  uint32_t data_size;
  uint64_t pc;
};

class OpuWarpinst {
public:
  uint64_t pc;
  uint32_t warp_id;
  opu_op_t op;
  uint32_t data_size;
  int vectorLength;
  opu_datatype_t data_type;
  opu_atomic_t m_atomic_spec;
  opu_cacheop_t cache_op;
  opu_mspace_t space_type;
  opu_memop_t memory_op;

  virtual uint64_t get_addr(uint32_t lane) const = 0;
  virtual uint64_t get_data(uint32_t lane) const = 0;
  virtual bool active(uint32_t lane) const = 0;
  virtual bool valid() const = 0;
  virtual bool empty() const = 0;

  virtual uint32_t active_count() const = 0;
  virtual uint32_t warp_size() const = 0;
  virtual bool is_load() const = 0;
  virtual bool is_store() const = 0;

  virtual bool isatomic() const = 0;
  opu_atomic_t get_atomic() const { return m_atomic_spec; }

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
