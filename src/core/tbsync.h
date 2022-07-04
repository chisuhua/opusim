#pragma once
#include "abstract_core.h"
#include "simt_common.h"

namespace opu {

class simt_core_ctx;

class barrier_set_t {
public:
  barrier_set_t(simt_core_ctx *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);

  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  // warp reaches exit
  void warp_exit(unsigned warp_id);

  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  unsigned m_max_warps_per_core;
  unsigned m_max_barriers_per_cta;
  unsigned m_warp_size;
  cta_to_warp_t m_cta_to_warps;
  bar_id_to_warp_t m_bar_id_to_warps;
  warp_set_t m_warp_active;
  warp_set_t m_warp_at_barrier;
  simt_core_ctx *m_shader;
};

struct insn_latency_info {
  unsigned pc;
  unsigned long latency;
};

}
