#ifndef __OPUCORE_BASE_HH__
#define __OPUCORE_BASE_HH__

class OpuSimBase;

class OpuCoreBase {
 public:
  explicit OpuCoreBase(class OpuSimBase *gpu);
  virtual void cycle() = 0;
  virtual void accept_fetch_response(mem_fetch *mf) = 0;
  virtual bool ldst_unit_wb_inst(warp_inst_t &inst) = 0;

  virtual void writeRegister(const warp_inst_t &inst, unsigned warpSize, unsigned lane_id, char* data) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;

  virtual void warp_reaches_barrier(warp_inst_t &inst) = 0;

  virtual bool fence_unblock_needed(unsigned warp_id) = 0;
  virtual void complete_fence(unsigned warp_id) = 0;

  virtual void finish_kernel() = 0;
};
#endif
