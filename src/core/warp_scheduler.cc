#include "simt_common.h"
#include "simt_core.h"
#include "warp_scheduler.h"
#include "warp_executor.h"
#include "opustats.h"
#include <algorithm>

warp_exec_t &scheduler_unit::warp(int i) { return *((*m_warp)[i]); }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

/**
 * A general function to order things in a Loose Round Robin way. The simplist use of this
 * function would be to implement a loose RR scheduler between all the warps assigned to this core.
 * A more sophisticated usage would be to order a set of "fetch groups" in a RR fashion.
 * In the first case, the templated class variable would be a simple unsigned int representing the
 * warp_id.  In the 2lvl case, T could be a struct or a list representing a set of warp_ids.
 * @param result_list: The resultant list the caller wants returned.  This list is cleared and then populated
 *                     in a loose round robin way
 * @param input_list: The list of things that should be put into the result_list. For a simple scheduler
 *                    this can simply be the m_supervised_warps list.
 * @param last_issued_from_input:  An iterator pointing the last member in the input_list that issued.
 *                                 Since this function orders in a RR fashion, the object pointed
 *                                 to by this iterator will be last in the prioritization list
 * @param num_warps_to_add: The number of warps you want the scheudler to pick between this cycle.
 *                          Normally, this will be all the warps availible on the core, i.e.
 *                          m_supervised_warps.size(). However, a more sophisticated scheduler may wish to
 *                          limit this number. If the number if < m_supervised_warps.size(), then only
 *                          the warps with highest RR priority will be placed in the result_list.
 */
template <class T>
void scheduler_unit::order_lrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  assert(num_warps_to_add <= input_list.size());
  result_list.clear();
  typename std::vector<T>::const_iterator iter =
      (last_issued_from_input == input_list.end()) ? input_list.begin()
                                                   : last_issued_from_input + 1;

  for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
    if (iter == input_list.end()) {
      iter = input_list.begin();
    }
    result_list.push_back(*iter);
  }
}

template <class T>
void scheduler_unit::order_rrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  result_list.clear();

  if (m_num_issued_last_cycle > 0 || warp(m_current_turn_warp).done_exit() ||
      warp(m_current_turn_warp).waiting()) {
    std::vector<warp_exec_t *>::const_iterator iter =
      (last_issued_from_input == input_list.end()) ? 
        input_list.begin() : last_issued_from_input + 1;
    for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
      if (iter == input_list.end()) {
      iter = input_list.begin();
      }
      unsigned warp_id = (*iter)->get_warp_id();
      if (!(*iter)->done_exit() && !(*iter)->waiting()) {
        result_list.push_back(*iter);
        m_current_turn_warp = warp_id;
        break;
      }
    }
  } else {
    result_list.push_back(&warp(m_current_turn_warp));
  }
}
/**
 * A general function to order things in an priority-based way.
 * The core usage of the function is similar to order_lrr.
 * The explanation of the additional parameters (beyond order_lrr) explains the further extensions.
 * @param ordering: An enum that determines how the age function will be treated in prioritization
 *                  see the definition of OrderingType.
 * @param priority_function: This function is used to sort the input_list.  It is passed to stl::sort as
 *                           the sorting fucntion. So, if you wanted to sort a list of integer warp_ids
 *                           with the oldest warps having the most priority, then the priority_function
 *                           would compare the age of the two warps.
 */
template <class T>
void scheduler_unit::order_by_priority(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add, OrderingType ordering,
    bool (*priority_func)(T lhs, T rhs)) {
  assert(num_warps_to_add <= input_list.size());
  result_list.clear();
  typename std::vector<T> temp = input_list;

  if (ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering) {
    T greedy_value = *last_issued_from_input;
    result_list.push_back(greedy_value);

    std::sort(temp.begin(), temp.end(), priority_func);
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      if (*iter != greedy_value) {
        result_list.push_back(*iter);
      }
    }
  } else if (ORDERED_PRIORITY_FUNC_ONLY == ordering) {
    std::sort(temp.begin(), temp.end(), priority_func);
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      result_list.push_back(*iter);
    }
  } else {
    fprintf(stderr, "Unknown ordering - %d\n", ordering);
    abort();
  }
}

void scheduler_unit::cycle() {
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  bool issued_inst = false;  // of these we issued one

  order_warps();
  for (std::vector<warp_exec_t *>::const_iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end(); iter++) {
    // Don't consider warps that are not yet valid
    if ((*iter) == NULL || (*iter)->done_exit()) {
      continue;
    }
    SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n",
                  (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    unsigned warp_id = (*iter)->get_warp_id();
    unsigned checked = 0;
    unsigned issued = 0;
    exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE;
    unsigned max_issue = m_shader->m_config->opu_max_insn_issue_per_warp;
    bool diff_exec_units =
        m_shader->m_config
            ->opu_dual_issue_diff_exec_units;  // In tis mode, we only allow
                                                 // dual issue to diff execution
                                                 // units (as in Maxwell and
                                                 // Pascal)

    if (warp(warp_id).ibuffer_empty())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    if (warp(warp_id).waiting())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for "
          "barrier\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() &&
           (checked < max_issue) && (checked <= issued) &&
           (issued < max_issue)) {
      const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
      // Jin: handle cdp latency;
#if 0
      if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
        assert(warp(warp_id).m_cdp_dummy);
        warp(warp_id).m_cdp_latency--;
        break;
      }
#endif
      bool valid = warp(warp_id).ibuffer_next_valid();
      bool warp_inst_issued = false;
      unsigned pc, rpc;
      m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
          m_shader->m_config->opu_ctx->func_sim->ptx_get_insn_str(pc)
              .c_str());
      if (pI) {
        assert(valid);
        if (pc != pI->pc) {
          SCHED_DPRINTF(
              "Warp (warp_id %u, dynamic_warp_id %u) control hazard (pc %u, pI->pc %u)"
              "instruction flush\n",
              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), pc, pI->pc);
          // control hazard
          warp(warp_id).set_next_pc(pc);
          warp(warp_id).ibuffer_flush();
        } else {
          valid_inst = true;
          if (!m_scoreboard->checkCollision(warp_id, pI)) {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
            ready_inst = true;

            const active_mask_t &active_mask =
                m_shader->get_active_mask(warp_id, pI);

            assert(warp(warp_id).inst_in_pipeline());

            if ((pI->op == opu_op_t::LOAD_OP) || (pI->op == opu_op_t::STORE_OP) ||
                (pI->op == opu_op_t::MEMORY_BARRIER_OP) ||
                (pI->op == opu_op_t::TENSOR_LOAD_OP) ||
                (pI->op == opu_op_t::TENSOR_STORE_OP)) {
              if (m_mem_out->has_free(m_shader->m_config->sub_core_model,
                                      m_id) &&
                  (!diff_exec_units ||
                   previous_issued_inst_exec_type != exec_unit_type_t::MEM)) {
                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id,
                                     m_id);
                issued++;
                issued_inst = true;
                warp_inst_issued = true;
                previous_issued_inst_exec_type = exec_unit_type_t::MEM;
              }
            } else {
              // This code need to be refactored
              if (pI->op != opu_op_t::TENSOR_OP && pI->op != opu_op_t::SFU_OP &&
                  pI->op != opu_op_t::DP_OP /*&& !(pI->op >= opu_op_t::SPEC_UNIT_START_ID)*/) {
                bool execute_on_SP = false;
                bool execute_on_INT = false;

                bool sp_pipe_avail =
                    (m_shader->m_config->opu_num_sp_units > 0) &&
                    m_sp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);
                bool int_pipe_avail =
                    (m_shader->m_config->opu_num_int_units > 0) &&
                    m_int_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                // if INT unit pipline exist, then execute ALU and INT
                // operations on INT unit and SP-FPU on SP unit (like in Volta)
                // if INT unit pipline does not exist, then execute all ALU, INT
                // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                if (m_shader->m_config->opu_num_int_units > 0 &&
                    int_pipe_avail && pI->op != opu_op_t::SP_OP &&
                    !(diff_exec_units &&
                      previous_issued_inst_exec_type == exec_unit_type_t::INT))
                  execute_on_INT = true;
                else if (sp_pipe_avail &&
                         (m_shader->m_config->opu_num_int_units == 0 ||
                          (m_shader->m_config->opu_num_int_units > 0 &&
                           pI->op == opu_op_t::SP_OP)) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::SP))
                  execute_on_SP = true;

                if (execute_on_INT || execute_on_SP) {
                  // Jin: special for CDP api
#if 0
                  if (pI->m_is_cdp && !warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);

                    if (pI->m_is_cdp == 1)
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->opu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1];
                    else  // cudaLaunchDeviceV2 and cudaGetParameterBufferV2
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->opu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1] +
                          m_shader->m_config->opu_ctx->func_sim
                                  ->cdp_latency[pI->m_is_cdp] *
                              active_mask.count();
                    warp(warp_id).m_cdp_dummy = true;
                    break;
                  } else if (pI->m_is_cdp && warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);
                    warp(warp_id).m_cdp_dummy = false;
                  }
#endif
                }

                if (execute_on_SP) {
                  m_shader->issue_warp(*m_sp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SP;
                } else if (execute_on_INT) {
                  m_shader->issue_warp(*m_int_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::INT;
                }
              } else if ((m_shader->m_config->opu_num_dp_units > 0) &&
                         (pI->op == opu_op_t::DP_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::DP)) {
                bool dp_pipe_avail =
                    (m_shader->m_config->opu_num_dp_units > 0) &&
                    m_dp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);

                if (dp_pipe_avail) {
                  m_shader->issue_warp(*m_dp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::DP;
                }
              }  // If the DP units = 0 (like in Fermi archi), then execute DP
                 // inst on SFU unit
              else if (((m_shader->m_config->opu_num_dp_units == 0 &&
                         pI->op == opu_op_t::DP_OP) ||
                        (pI->op == opu_op_t::SFU_OP) || (pI->op == opu_op_t::ALU_SFU_OP)) &&
                       !(diff_exec_units && previous_issued_inst_exec_type ==
                                                exec_unit_type_t::SFU)) {
                bool sfu_pipe_avail =
                    (m_shader->m_config->opu_num_sfu_units > 0) &&
                    m_sfu_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                if (sfu_pipe_avail) {
                  m_shader->issue_warp(*m_sfu_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SFU;
                }
              } else if ((pI->op == opu_op_t::TENSOR_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::TENSOR)) {
                bool tensor_core_pipe_avail =
                    (m_shader->m_config->opu_num_tensor_core_units > 0) &&
                    m_tensor_core_out->has_free(
                        m_shader->m_config->sub_core_model, m_id);

                if (tensor_core_pipe_avail) {
                  m_shader->issue_warp(*m_tensor_core_out, pI, active_mask,
                                       warp_id, m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;
                }
              }/* else if ((pI->op >= SPEC_UNIT_START_ID) &&
                         !(diff_exec_units &&
                           previous_issued_inst_exec_type ==
                               exec_unit_type_t::SPECIALIZED)) {
                unsigned spec_id = pI->op - SPEC_UNIT_START_ID;
                assert(spec_id < m_shader->m_config->m_specialized_unit.size());
                register_set *spec_reg_set = m_spec_cores_out[spec_id];
                bool spec_pipe_avail =
                    (m_shader->m_config->m_specialized_unit[spec_id].num_units >
                     0) &&
                    spec_reg_set->has_free(m_shader->m_config->sub_core_model,
                                           m_id);

                if (spec_pipe_avail) {
                  m_shader->issue_warp(*spec_reg_set, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type =
                      exec_unit_type_t::SPECIALIZED;
                }
              }*/

            }  // end of else
          } else {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          }
        }
      } else if (valid) {
        // this case can happen after a return instruction in diverged warp
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp "
            "flush\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
        warp(warp_id).set_next_pc(pc);
        warp(warp_id).ibuffer_flush();
      }
      if (warp_inst_issued) {
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), issued);
        do_on_warp_issued(warp_id, issued, iter);
      }
      checked++;
    }
    if (issued) {
      // This might be a bit inefficient, but we need to maintain
      // two ordered list for proper scheduler execution.
      // We could remove the need for this loop by associating a
      // supervised_is index with each entry in the
      // m_next_cycle_prioritized_warps vector. For now, just run through until
      // you find the right warp_id
      for (std::vector<warp_exec_t *>::const_iterator supervised_iter =
               m_supervised_warps.begin();
           supervised_iter != m_supervised_warps.end(); ++supervised_iter) {
        if (*iter == *supervised_iter) {
          m_last_supervised_issued = supervised_iter;
        }
      }
      m_num_issued_last_cycle = issued;
      if (issued == 1)
        m_stats->single_issue_nums[m_id]++;
      else if (issued > 1)
        m_stats->dual_issue_nums[m_id]++;
      else
        abort();  // issued should be > 0

      break;
    }
  }

  // issue stall statistics:
  if (!valid_inst)
    m_stats->shader_cycle_distro[0]++;  // idle or control hazard
  else if (!ready_inst)
    m_stats->shader_cycle_distro[1]++;  // waiting for RAW hazards (possibly due
                                        // to memory)
  else if (!issued_inst)
    m_stats->shader_cycle_distro[2]++;  // pipeline stalled
}

void scheduler_unit::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<warp_exec_t *>::const_iterator &prioritized_iter) {
  m_stats->event_warp_issued(m_shader->get_sid(), warp_id, num_issued,
                             warp(warp_id).get_dynamic_warp_id());
  warp(warp_id).ibuffer_step();
}

bool scheduler_unit::sort_warps_by_oldest_dynamic_id(warp_exec_t *lhs,
                                                     warp_exec_t *rhs) {
  if (rhs && lhs) {
    if (lhs->done_exit() || lhs->waiting()) {
      return false;
    } else if (rhs->done_exit() || rhs->waiting()) {
      return true;
    } else {
      return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
    }
  } else {
    return lhs < rhs;
  }
}

void lrr_scheduler::order_warps() {
  order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}
void rrr_scheduler::order_warps() {
  order_rrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}

void gto_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}

void oldest_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERED_PRIORITY_FUNC_ONLY,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}

void two_level_active_scheduler::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<warp_exec_t *>::const_iterator &prioritized_iter) {
  scheduler_unit::do_on_warp_issued(warp_id, num_issued, prioritized_iter);
  if (SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization) {
    std::vector<warp_exec_t *> new_active;
    order_lrr(new_active, m_next_cycle_prioritized_warps, prioritized_iter,
              m_next_cycle_prioritized_warps.size());
    m_next_cycle_prioritized_warps = new_active;
  } else {
    fprintf(stderr, "Unimplemented m_inner_level_prioritization: %d\n",
            m_inner_level_prioritization);
    abort();
  }
}

void two_level_active_scheduler::order_warps() {
  // Move waiting warps to m_pending_warps
  unsigned num_demoted = 0;
  for (std::vector<warp_exec_t *>::iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end();) {
    bool waiting = (*iter)->waiting();
    for (int i = 0; i < MAX_INPUT_VALUES; i++) {
      const warp_inst_t *inst = (*iter)->ibuffer_next_inst();
      // Is the instruction waiting on a long operation?
      if (inst && inst->in(i) > 0 &&
          this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in(i))) {
        waiting = true;
      }
    }

    if (waiting) {
      m_pending_warps.push_back(*iter);
      iter = m_next_cycle_prioritized_warps.erase(iter);
      SCHED_DPRINTF("DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                    (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
      ++num_demoted;
    } else {
      ++iter;
    }
  }

  // If there is space in m_next_cycle_prioritized_warps, promote the next
  // m_pending_warps
  unsigned num_promoted = 0;
  if (SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization) {
    while (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
      m_pending_warps.pop_front();
      SCHED_DPRINTF(
          "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
          (m_next_cycle_prioritized_warps.back())->get_warp_id(),
          (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id());
      ++num_promoted;
    }
  } else {
    fprintf(stderr, "Unimplemented m_outer_level_prioritization: %d\n",
            m_outer_level_prioritization);
    abort();
  }
  assert(num_promoted == num_demoted);
}

swl_scheduler::swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<warp_exec_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_string)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out, mem_out,
                     id) {
  unsigned m_prioritization_readin;
  int ret = sscanf(config_string, "warp_limiting:%d:%d",
                   &m_prioritization_readin, &m_num_warps_to_limit);
  assert(2 == ret);
  m_prioritization = (scheduler_prioritization_type)m_prioritization_readin;
  // Currently only GTO is implemented
  assert(m_prioritization == SCHEDULER_PRIORITIZATION_GTO);
  assert(m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader);
}

void swl_scheduler::order_warps() {
  if (SCHEDULER_PRIORITIZATION_GTO == m_prioritization) {
    order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                      m_last_supervised_issued,
                      MIN(m_num_warps_to_limit, m_supervised_warps.size()),
                      ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                      scheduler_unit::sort_warps_by_oldest_dynamic_id);
  } else {
    fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
    abort();
  }
}

