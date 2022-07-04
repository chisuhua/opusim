#include "mem_fetch.h"

namespace opu {
unsigned mem_fetch::sm_next_mf_request_uid = 1;

mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc,
                     unsigned long long cycle, mem_fetch *m_original_mf,
                     mem_fetch *m_original_wr_mf)
    : m_access(access)

{
  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
}

mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

// FIXME void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

// bool mem_fetch::istexture() const {
//  if (m_inst.empty()) return false;
//  return m_inst.space.get_type() == tex_space;
//}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.get_space() == opu_mspace_t::CONST) ||
         (m_inst.get_space() == opu_mspace_t::PARAM);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}
}
