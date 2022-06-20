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

  virtual uint64_t get_addr(uint32_t lane) const {};
  virtual uint64_t get_data(uint32_t lane) const {};
  virtual bool active(uint32_t lane) const {};
  virtual bool valid() const { return valid_;};
  virtual bool empty() const {};
  virtual opu_mspace_t get_space() const { return space_type; };

  virtual uint32_t active_count() const {};
  virtual uint32_t warp_size() const { return warp_size_;};
  virtual bool is_load() const {};
  virtual bool is_store() const {};

  virtual bool isatomic() const {};
  opu_atomic_t get_atomic() const { return m_atomic_spec; }

  bool valid_;
  uint32_t warp_size_;
};

