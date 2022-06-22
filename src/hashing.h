#pragma once
#include <stdint.h>
#include <cassert>
#include "simt_common.h"

uint32_t ipoly_hash_function(address_type higher_bits, uint32_t index,
                             uint32_t bank_set_num);

uint32_t bitwise_hash_function(address_type higher_bits, uint32_t index,
                               uint32_t bank_set_num);

uint32_t PAE_hash_function(address_type higher_bits, uint32_t index,
                           uint32_t bank_set_num);


