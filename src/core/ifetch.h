#pragma once

#include "simt_common.h"
#include <memory>

namespace opu {

struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id, uint8_t *ptr) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
    std::memcpy(&opcode[0], ptr, 16);
  }

  bool m_valid;
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
  uint8_t opcode[16];
};

}
