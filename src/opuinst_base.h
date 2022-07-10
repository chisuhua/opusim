#pragma once
#include "coasm.h"
#include <stdint.h>

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
  virtual const uint8_t *get_data(uint32_t lane) const = 0;
  virtual bool active(uint32_t lane) const = 0;
  virtual bool valid() const = 0;
  virtual bool empty() const = 0;
  virtual opu_mspace_t get_space() const { return space_type; };

  virtual uint32_t active_count() const = 0;
  virtual uint32_t warp_size() const = 0;
  virtual bool is_load() const {
    return (op == opu_op_t::LOAD_OP || op == opu_op_t::TENSOR_LOAD_OP ||
            memory_op == opu_memop_t::LOAD);

  };
  virtual bool is_store() const {
    return (op == opu_op_t::STORE_OP || op == opu_op_t::TENSOR_STORE_OP ||
            memory_op == opu_memop_t::STORE);
  };
  virtual bool isatomic() const = 0;
  opu_atomic_t get_atomic() const { return m_atomic_spec; }

  // uint32_t warp_size_ {MAX_WARP_SIZE};
};

