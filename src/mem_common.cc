#pragma once
#include "mem_common.h"
#include "mem_fetch.h"
#include "core/warp_inst.h"

namespace opu {
unsigned mem_access_t::sm_next_access_uid = 0;

// extern opu_sim *g_the_gpu;
mem_fetch *simt_core_mem_fetch_allocator::alloc(
    address_type addr, mem_access_type type, unsigned size, bool wr,
    unsigned long long cycle) const {
  mem_access_t access(type, addr, size, wr, m_ctx);
  mem_fetch *mf =
      new mem_fetch(access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1,
                    m_core_id, m_cluster_id, cycle);
  return mf;
}

mem_fetch *simt_core_mem_fetch_allocator::alloc(
    address_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
          m_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid, m_core_id,
      m_cluster_id, cycle, original_mf);
  return mf;
}

mem_fetch *simt_core_mem_fetch_allocator::alloc(
        const warp_inst_t &inst, const mem_access_t &access,
        unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy,
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, cycle);
    return mf;
}

}
