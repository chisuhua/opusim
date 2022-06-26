#include "warp_inst.h"
#include "inc/Instruction.h"
#include "opuconfig.h"

address_type line_size_based_tag_func(address_type address,
                                       address_type line_size) {
  // gives the tag for an address based on a given line size
  return address & ~(line_size - 1);
}

warp_inst_t::warp_inst_t() {
    m_uid = 0;
    m_empty = true;
    m_isatomic = false;
    m_per_scalar_thread_valid = false;
    m_mem_accesses_created = false;
    m_cache_hit = false;
    should_do_atomic = true;
    m_config = shader_core_config::getInstance();
}

void warp_inst_t::issue(const active_mask_t &mask, unsigned warp_id,
                        unsigned long long cycle, int dynamic_warp_id,
                        int sch_id) {
  m_warp_active_mask = mask;
  m_warp_issued_mask = mask;
  // m_uid = ++(m_config->opu_ctx->warp_inst_sm_next_uid);
  m_warp_id = warp_id;
  m_dynamic_warp_id = dynamic_warp_id;
  issue_cycle = cycle;
  // cycles = initiation_interval;
  m_cache_hit = false;
  m_empty = false;
  m_scheduler_id = sch_id;
}

void move_warp(warp_inst_t *&dst, warp_inst_t *&src) {
  assert(dst->empty());
  warp_inst_t *temp = dst;
  dst = src;
  src = temp;
  src->clear();
}

void warp_inst_t::clear_active(const active_mask_t &inactive) {
  active_mask_t test = m_warp_active_mask;
  test &= inactive;
  assert(test == inactive);  // verify threads being disabled were active
  m_warp_active_mask &= ~inactive;
}

void warp_inst_t::set_not_active(unsigned lane_id) {
  m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active(const active_mask_t &active) {
  m_warp_active_mask = active;
  if (m_isatomic) {
    for (unsigned i = 0; i < MAX_WARP_SIZE; i++) {
      if (!m_warp_active_mask.test(i)) {
        // m_per_scalar_thread[i].callback.function = NULL;
        // m_per_scalar_thread[i].callback.instruction = NULL;
        // m_per_scalar_thread[i].callback.thread = NULL;
      }
    }
  }
}

void warp_inst_t::Execute(WarpState* warp_state, uint32_t lane) {
  return m_instruction->Execute(warp_state, lane);
}

uint32_t warp_inst_t::GetSize() const {
  return m_instruction->GetSize();
}

uint32_t warp_inst_t::get_num_operands() const {
  return m_instruction->get_num_regs();
}

uint32_t warp_inst_t::get_num_regs()  const {
  return m_instruction->get_num_regs();
}

int32_t warp_inst_t::out(uint32_t t) const {
  int32_t reg_idx = m_instruction->operands_[Operand::DST]->getRegIdx(t);
}

int32_t warp_inst_t::outcount() const {
  int32_t range = 0;
  for (int dst = 0; dst <= m_instruction->num_dst_operands; dst++) {
    range += m_instruction->operands_[Operand::SRC0]->reg_.range_;
  }
  return range;
}

int32_t warp_inst_t::in(uint32_t t) const {
  int32_t reg_idx = -1;
  int32_t reg0_idx = m_instruction->operands_[Operand::SRC0]->reg_.reg_idx_;
  int32_t reg1_idx = m_instruction->operands_[Operand::SRC1]->reg_.reg_idx_;
  int32_t reg2_idx = m_instruction->operands_[Operand::SRC2]->reg_.reg_idx_;
  int32_t reg3_idx = m_instruction->operands_[Operand::SRC3]->reg_.reg_idx_;
  int32_t reg0_range = m_instruction->operands_[Operand::SRC0]->reg_.range_;
  int32_t reg1_range = m_instruction->operands_[Operand::SRC1]->reg_.range_;
  int32_t reg2_range = m_instruction->operands_[Operand::SRC2]->reg_.range_;
  int32_t reg3_range = m_instruction->operands_[Operand::SRC3]->reg_.range_;

  if (t < reg0_range) { return reg0_idx + t;}
  t -= reg0_range;
  if (t < reg1_range) { return reg1_idx + t;}
  t -= reg1_range;
  if (t < reg2_range) { return reg2_idx + t;}
  t -= reg2_range;
  if (t < reg3_range) { return reg3_idx + t;}
  return -1;
}

int32_t warp_inst_t::incount() const {
  int32_t range = 0;
  for (int src = 0; src <= m_instruction->num_src_operands; src++) {
    range += m_instruction->operands_[src]->reg_.range_;
  }
  return range;
}

#if 0
void warp_inst_t::do_atomic(bool forceDo) {
  do_atomic(m_warp_active_mask, forceDo);
}

void warp_inst_t::do_atomic(const active_mask_t &access_mask, bool forceDo) {
  assert(m_isatomic && (!m_empty || forceDo));
  if (!should_do_atomic) return;
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    if (access_mask.test(i)) {
      dram_callback_t &cb = m_per_scalar_thread[i].callback;
      if (cb.thread) cb.function(cb.instruction, cb.thread);
    }
  }
}
#endif

void warp_inst_t::broadcast_barrier_reduction(
    const active_mask_t &access_mask) {
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    if (access_mask.test(i)) {
#if 0
        FIXME
      dram_callback_t &cb = m_per_scalar_thread[i].callback;
      if (cb.thread) {
        cb.function(cb.instruction, cb.thread);
      }
#endif
    }
  }
}

void warp_inst_t::generate_mem_accesses() {
  if (empty() || op == opu_op_t::MEMORY_BARRIER_OP || m_mem_accesses_created) return;
  if (!((op == opu_op_t::LOAD_OP) || (op == opu_op_t::TENSOR_LOAD_OP) || (op == opu_op_t::STORE_OP) ||
        (op == opu_op_t::TENSOR_STORE_OP) ))
    return;
  if (m_warp_active_mask.count() == 0) return;  // predicated off
    // In gem5-gpu, global, const and local references go through the gem5-gpu LSQ
    if ( get_space() == opu_mspace_t::GLOBAL || get_space() == opu_mspace_t::CONST || get_space() == opu_mspace_t::PRIVATE )
        return;

  const size_t starting_queue_size = m_accessq.size();

  assert(is_load() || is_store());

  //if((space.get_type() != tex_space) && (space.get_type() != const_space))
    assert(m_per_scalar_thread_valid);  // need address information per thread

  bool is_write = is_store();

  mem_access_type access_type;
  switch (get_space()) {
      case opu_mspace_t::CONST:
      case opu_mspace_t::PARAM:
      access_type = CONST_ACC_R;
      break;
      /*
      case opu_mspace_t::tex_space:
      access_type = TEXTURE_ACC_R;
      break;
      */
      case opu_mspace_t::GLOBAL:
      access_type = is_write ? GLOBAL_ACC_W : GLOBAL_ACC_R;
      break;
      case opu_mspace_t::PRIVATE:
      // case opu_mspace_t::param_space_local:
      access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      break;
      case opu_mspace_t::SHARED:
      break;
    default:
      assert(0);
      break;
  }

  // Calculate memory accesses generated by this warp
  address_type cache_block_size = 0;  // in bytes

  switch (get_space()) {
      case opu_mspace_t::SHARED:
    /*case sstarr_space:*/ {
      unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
      unsigned total_accesses = 0;
      for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;
           subwarp++) {
        // data structures used per part warp
        std::map<unsigned, std::map<address_type, unsigned> >
            bank_accs;  // bank -> word address -> access count

        // step 1: compute accesses to words in banks
        for (unsigned thread = subwarp * subwarp_size;
             thread < (subwarp + 1) * subwarp_size; thread++) {
          if (!active(thread)) continue;
          address_type addr = m_per_scalar_thread[thread].memreqaddr[0];
          // FIXME: deferred allocation of shared memory should not accumulate
          // across kernel launches assert( addr < m_config->opu_shmem_size );
          unsigned bank = m_config->shmem_bank_func(addr);
          address_type word =
              line_size_based_tag_func(addr, m_config->WORD_SIZE);
          bank_accs[bank][word]++;
        }

        if (m_config->shmem_limited_broadcast) {
          // step 2: look for and select a broadcast bank/word if one occurs
          bool broadcast_detected = false;
          address_type broadcast_word = (address_type)-1;
          unsigned broadcast_bank = (unsigned)-1;
          std::map<unsigned, std::map<address_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            unsigned bank = b->first;
            std::map<address_type, unsigned> &access_set = b->second;
            std::map<address_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w) {
              if (w->second > 1) {
                // found a broadcast
                broadcast_detected = true;
                broadcast_bank = bank;
                broadcast_word = w->first;
                break;
              }
            }
            if (broadcast_detected) break;
          }
          // step 3: figure out max bank accesses performed, taking account of
          // broadcast case
          unsigned max_bank_accesses = 0;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            unsigned bank_accesses = 0;
            std::map<address_type, unsigned> &access_set = b->second;
            std::map<address_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w)
              bank_accesses += w->second;
            if (broadcast_detected && broadcast_bank == b->first) {
              for (w = access_set.begin(); w != access_set.end(); ++w) {
                if (w->first == broadcast_word) {
                  unsigned n = w->second;
                  assert(n > 1);  // or this wasn't a broadcast
                  assert(bank_accesses >= (n - 1));
                  bank_accesses -= (n - 1);
                  break;
                }
              }
            }
            if (bank_accesses > max_bank_accesses)
              max_bank_accesses = bank_accesses;
          }

          // step 4: accumulate
          total_accesses += max_bank_accesses;
        } else {
          // step 2: look for the bank with the maximum number of access to
          // different words
          unsigned max_bank_accesses = 0;
          std::map<unsigned, std::map<address_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            max_bank_accesses =
                std::max(max_bank_accesses, (unsigned)b->second.size());
          }

          // step 3: accumulate
          total_accesses += max_bank_accesses;
        }
      }
      assert(total_accesses > 0 && total_accesses <= m_config->warp_size);
      cycles = total_accesses;  // shared memory conflicts modeled as larger
                                // initiation interval
      // m_config->opu_ctx->stats->ptx_file_line_stats_add_smem_bank_conflict(
      //    pc, total_accesses);
      break;
    }
/*
    case tex_space:
      cache_block_size = m_config->opu_cache_texl1_linesize;
      break;
      */
    case opu_mspace_t::CONST:
    case opu_mspace_t::PARAM:
      cache_block_size = m_config->opu_cache_constl1_linesize;
      break;

    case opu_mspace_t::GLOBAL:
    case opu_mspace_t::PRIVATE:
    // case param_space_local:
      if (m_config->opu_coalesce_arch >= 13) {
          /* FIXME
        if (isatomic())
          memory_coalescing_arch_atomic(is_write, access_type);
        else
        */
          memory_coalescing_arch(is_write, access_type);
      } else
        abort();

      break;

    default:
      abort();
  }

  if (cache_block_size) {
    assert(m_accessq.empty());
    mem_access_byte_mask_t byte_mask;
    std::map<address_type, active_mask_t>
        accesses;  // block address -> set of thread offsets in warp
    std::map<address_type, active_mask_t>::iterator a;
    for (unsigned thread = 0; thread < m_config->warp_size; thread++) {
      if (!active(thread)) continue;
      address_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      address_type block_address =
          line_size_based_tag_func(addr, cache_block_size);
      accesses[block_address].set(thread);
      unsigned idx = addr - block_address;
      for (unsigned i = 0; i < data_size; i++) byte_mask.set(idx + i);
    }
    for (a = accesses.begin(); a != accesses.end(); ++a)
      m_accessq.push_back(mem_access_t(
          access_type, a->first, cache_block_size, is_write, a->second,
          byte_mask, mem_access_sector_mask_t(), m_config->opu_ctx));
  }
/*
  if (get_space() == opu_mspace_t::GLOBAL) {
    m_config->opu_ctx->stats->ptx_file_line_stats_add_uncoalesced_gmem(
        pc, m_accessq.size() - starting_queue_size);
  }
  */
  m_mem_accesses_created = true;
}

void warp_inst_t::memory_coalescing_arch(bool is_write,
                                         mem_access_type access_type) {
  // see the CUDA manual where it discusses coalescing rules before reading this
#if 0
  unsigned segment_size = 0;
  unsigned warp_parts = m_config->mem_warp_parts;
  bool sector_segment_size = false;

  if (m_config->opu_coalesce_arch >= 20 &&
      m_config->opu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;
    else
      sector_segment_size = false;
  } else if (m_config->opu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }

  switch (data_size) {
    case 1:
      segment_size = 32;
      break;
    case 2:
      segment_size = sector_segment_size ? 32 : 64;
      break;
    case 4:
    case 8:
    case 16:
      segment_size = sector_segment_size ? 32 : 128;
      break;
  }
  unsigned subwarp_size = m_config->warp_size / warp_parts;

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    std::map<address_type, transaction_info> subwarp_transactions;

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      if (!active(thread)) continue;

      unsigned data_size_coales = data_size;
      unsigned num_accesses = 1;

      if (space.get_type() == local_space ||
          space.get_type() == param_space_local) {
        // Local memory accesses >4B were split into 4B chunks
        if (data_size >= 4) {
          data_size_coales = 4;
          num_accesses = data_size / 4;
        }
        // Otherwise keep the same data_size for sub-4B access to local memory
      }

      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

      //            for(unsigned access=0; access<num_accesses; access++) {
      for (unsigned access = 0;
           (access < MAX_ACCESSES_PER_INSN_PER_THREAD) &&
           (m_per_scalar_thread[thread].memreqaddr[access] != 0);
           access++) {
        address_type addr = m_per_scalar_thread[thread].memreqaddr[access];
        address_type block_address =
            line_size_based_tag_func(addr, segment_size);
        unsigned chunk =
            (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte
                                // chunk does this thread access?
        transaction_info &info = subwarp_transactions[block_address];

        // can only write to one segment
        // it seems like in trace driven, a thread can write to more than one
        // segment assert(block_address ==
        // line_size_based_tag_func(addr+data_size_coales-1,segment_size));

        info.chunks.set(chunk);
        info.active.set(thread);
        unsigned idx = (addr & 127);
        for (unsigned i = 0; i < data_size_coales; i++)
          if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);

        // it seems like in trace driven, a thread can write to more than one
        // segment handle this special case
        if (block_address != line_size_based_tag_func(
                                 addr + data_size_coales - 1, segment_size)) {
          addr = addr + data_size_coales - 1;
          address_type block_address =
              line_size_based_tag_func(addr, segment_size);
          unsigned chunk = (addr & 127) / 32;
          transaction_info &info = subwarp_transactions[block_address];
          info.chunks.set(chunk);
          info.active.set(thread);
          unsigned idx = (addr & 127);
          for (unsigned i = 0; i < data_size_coales; i++)
            if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);
        }
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<address_type, transaction_info>::iterator t;
    for (t = subwarp_transactions.begin(); t != subwarp_transactions.end();
         t++) {
      address_type addr = t->first;
      const transaction_info &info = t->second;

      memory_coalescing_arch_reduce_and_send(is_write, access_type, info, addr,
                                             segment_size);
    }
  }
#endif
}

#if 0
void warp_inst_t::memory_coalescing_arch_atomic(bool is_write,
                                                mem_access_type access_type) {
  assert(space.get_type() ==
         global_space);  // Atomics allowed only for global memory

  // see the CUDA manual where it discusses coalescing rules before reading this
  unsigned segment_size = 0;
  unsigned warp_parts = m_config->mem_warp_parts;
  bool sector_segment_size = false;

  if (m_config->opu_coalesce_arch >= 20 &&
      m_config->opu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;
    else
      sector_segment_size = false;
  } else if (m_config->opu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }

   switch( data_size ) {
   case 1: segment_size = 32; break;
   case 2: segment_size = sector_segment_size? 32 : 64; break;
   case 4: case 8: case 16: segment_size = sector_segment_size? 32 : 128; break;
   }
  unsigned subwarp_size = m_config->warp_size / warp_parts;

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    std::map<address_type, std::list<transaction_info> >
        subwarp_transactions;  // each block addr maps to a list of transactions

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      if (!active(thread)) continue;

      address_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      address_type block_address =
          line_size_based_tag_func(addr, segment_size);
      unsigned chunk =
          (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte chunk
                              // does this thread access?

      // can only write to one segment
      assert(block_address ==
             line_size_based_tag_func(addr + data_size - 1, segment_size));

      // Find a transaction that does not conflict with this thread's accesses
      bool new_transaction = true;
      std::list<transaction_info>::iterator it;
      transaction_info *info;
      for (it = subwarp_transactions[block_address].begin();
           it != subwarp_transactions[block_address].end(); it++) {
        unsigned idx = (addr & 127);
        if (not it->test_bytes(idx, idx + data_size - 1)) {
          new_transaction = false;
          info = &(*it);
          break;
        }
      }
      if (new_transaction) {
        // Need a new transaction
        subwarp_transactions[block_address].push_back(transaction_info());
        info = &subwarp_transactions[block_address].back();
      }
      assert(info);

      info->chunks.set(chunk);
      info->active.set(thread);
      unsigned idx = (addr & 127);
      for (unsigned i = 0; i < data_size; i++) {
        assert(!info->bytes.test(idx + i));
        info->bytes.set(idx + i);
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<address_type, std::list<transaction_info> >::iterator t_list;
    for (t_list = subwarp_transactions.begin();
         t_list != subwarp_transactions.end(); t_list++) {
      // For each block addr
      address_type addr = t_list->first;
      const std::list<transaction_info> &transaction_list = t_list->second;

      std::list<transaction_info>::const_iterator t;
      for (t = transaction_list.begin(); t != transaction_list.end(); t++) {
        // For each transaction
        const transaction_info &info = *t;
        memory_coalescing_arch_reduce_and_send(is_write, access_type, info,
                                               addr, segment_size);
      }
    }
  }
}

void warp_inst_t::memory_coalescing_arch_reduce_and_send(
    bool is_write, mem_access_type access_type, const transaction_info &info,
    address_type addr, unsigned segment_size) {
  assert((addr & (segment_size - 1)) == 0);

  const std::bitset<4> &q = info.chunks;
  assert(q.count() >= 1);
  std::bitset<2> h;  // halves (used to check if 64 byte segment can be
                     // compressed into a single 32 byte segment)

  unsigned size = segment_size;
  if (segment_size == 128) {
    bool lower_half_used = q[0] || q[1];
    bool upper_half_used = q[2] || q[3];
    if (lower_half_used && !upper_half_used) {
      // only lower 64 bytes used
      size = 64;
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else if ((!lower_half_used) && upper_half_used) {
      // only upper 64 bytes used
      addr = addr + 64;
      size = 64;
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    } else {
      assert(lower_half_used && upper_half_used);
    }
  } else if (segment_size == 64) {
    // need to set halves
    if ((addr % 128) == 0) {
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else {
      assert((addr % 128) == 64);
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    }
  }
  if (size == 64) {
    bool lower_half_used = h[0];
    bool upper_half_used = h[1];
    if (lower_half_used && !upper_half_used) {
      size = 32;
    } else if ((!lower_half_used) && upper_half_used) {
      addr = addr + 32;
      size = 32;
    } else {
      assert(lower_half_used && upper_half_used);
    }
  }
  m_accessq.push_back(mem_access_t(access_type, addr, size, is_write,
                                   info.active, info.bytes, info.chunks,
                                   m_config->opu_ctx));
}
#endif
void warp_inst_t::completed(unsigned long long cycle) const {
  unsigned long long latency = cycle - issue_cycle;
  assert(latency <= cycle);  // underflow detection
  // m_config->opu_ctx->stats->ptx_file_line_stats_add_latency(
  //    pc, latency * active_count());
}

void warp_inst_t::print(FILE *fout) const {
  if (empty()) {
    fprintf(fout, "bubble\n");
    return;
  } else
    fprintf(fout, "0x%04x ", pc);
  fprintf(fout, "w%02d[", m_warp_id);
  for (unsigned j = 0; j < m_config->warp_size; j++)
    fprintf(fout, "%c", (active(j) ? '1' : '0'));
  fprintf(fout, "]: ");
  // m_config->opu_ctx->func_sim->ptx_print_insn(pc, fout);
  fprintf(fout, "\n");
}
