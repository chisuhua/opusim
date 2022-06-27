#include "abstract_core.h"
#include "simt_common.h"
#include "inc/Instruction.h"


void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId) {
  active_mask_t active_mask = warp_active_mask(warpId);
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (active_mask.test(t)) {
      // if (warpId == (unsigned(-1))) warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;
      get_warp_state(warpId)->incWarpPC(inst.GetSize(), t);
      inst.Execute(get_warp_state(warpId), t);
      // FIXME m_thread[tid]->ptx_exec_inst(inst, t);

      // virtual function
      // checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }
}

// TODO schi add
void core_t::writeRegister(const OpuWarpinst &inst_, unsigned warpSize, unsigned lane_id, char *data) {
    auto inst = dynamic_cast<warp_inst_t>(inst_);
    // assert(inst.active(lane_id));
    int warp_id = inst.warp_id();
    assert(warp_active_mask(warp_id).test(lane_id));
    // m_thread[warpSize*warpId+lane_id]->writeRegister(inst, lane_id, data);
    uint32_t reg_idx = inst.m_instruction->operands_[Operand::DST]->reg_.reg_idx_;
    get_warp_state(warp_id)->setVreg(reg_idx, *data, lane_id);
}

bool core_t::ptx_thread_done(uint32_t warp_id, unsigned lane_id) const {
  WarpState *warp_state = get_warp_state(warp_id);
  return warp_state->isLaneExit(lane_id);
}

void core_t::updateSIMTStack(unsigned warpId, warp_inst_t *inst) {
  simt_mask_t thread_done;
  addr_vector_t next_pc;
  WarpState *warp_state = get_warp_state(warpId);
  unsigned wtid = warpId * m_warp_size;
  for (unsigned i = 0; i < m_warp_size; i++) {
    if (ptx_thread_done(warpId, i)) {
      thread_done.set(i);
      next_pc.push_back((address_type)-1);
    } else {
      if (inst->m_instruction->reconvergence_pc == RECONVERGE_RETURN_PC)
        inst->m_instruction->reconvergence_pc = warp_state->get_return_pc(i);
      next_pc.push_back(warp_state->getThreadPC(i));
    }
  }
  m_simt_stack[warpId]->update(thread_done, next_pc, inst->m_instruction->reconvergence_pc,
                               inst->op, inst->GetSize(), inst->pc);
}

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId) {
  unsigned pc, rpc;
  m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
  // FIXME
  warp_inst_t wi; //FIXME *(m_gpu->gpgpu_ctx->ptx_fetch_inst(pc));
  wi.set_active(m_simt_stack[warpId]->get_active_mask());
  return wi;
}

void core_t::deleteSIMTStack() {
  if (m_simt_stack) {
    for (unsigned i = 0; i < m_warp_count; ++i) delete m_simt_stack[i];
    delete[] m_simt_stack;
    m_simt_stack = NULL;
  }
}

void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size) {
  m_simt_stack = new simt_stack *[warp_count];
  for (unsigned i = 0; i < warp_count; ++i)
    m_simt_stack[i] = new simt_stack(i, warp_size/*, m_gpu*/);
  m_warp_size = warp_size;
  m_warp_count = warp_count;
}

void core_t::get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                                     unsigned *rpc) const {
  m_simt_stack[warpId]->get_pdom_stack_top_info(pc, rpc);
}
