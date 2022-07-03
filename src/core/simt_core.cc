// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh 
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "simt_core.h"
#include "simt_core_cluster.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include "opu_sim.h"
// #include "../../libcuda_sim/opu_context.h"
// #include "../cuda-sim/cuda-sim.h"
// #include "../cuda-sim/ptx-stats.h"
// #include "../cuda-sim/ptx_sim.h"
// #include "../statwrapper.h"
// #include "addrdec.h"
// #include "dram.h"
// #include "gpu-misc.h"
// #include "gpu-cache_gem5.h"
// #include "gpu-sim.h"
// #include "icnt_wrapper.h"
// #include "gpu/gpgpu-sim/cuda_gpu.hh"
// #include "mem_fetch.h"
// #include "mem_latency_stat.h"
// #include "shader_trace.h"
// #include "stat-tool.h"
// #include "traffic_breakdown.h"
// #include "visualizer.h"
#include "opuconfig.h"
#include "opustats.h"
#include "warp_scheduler.h"
#include "warp_executor.h"
#include "warp_inst.h"
#include "int_unit.h"
#include "sp_unit.h"
#include "dp_unit.h"
#include "sf_unit.h"
#include "specialized_unit.h"
#include "tensor_unit.h"
#include "ldst_unit.h"
#include "inc/WarpState.h"
#define TMPHACK
#include "inc/KernelInfo.h"

#define PRIORITIZE_MSHR_OVER_WB 1

#if 0
// extern opu_sim *g_the_gpu;
mem_fetch *simt_core_mem_fetch_allocator::alloc(
    address_type addr, mem_access_type type, unsigned size, bool wr,
    unsigned long long cycle) const {
  mem_access_t access(type, addr, size, wr, m_memory_config->opu_ctx);
  mem_fetch *mf =
      new mem_fetch(access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1,
                    m_core_id, m_cluster_id, m_memory_config, cycle);
  return mf;
}

mem_fetch *simt_core_mem_fetch_allocator::alloc(
    address_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->opu_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid, m_core_id,
      m_cluster_id, m_memory_config, cycle, original_mf);
  return mf;
}
#endif
/////////////////////////////////////////////////////////////////////////////

std::list<unsigned> simt_core_ctx::get_regs_written(const warp_inst_t &fvt) const {
  std::list<unsigned> result;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    // int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
    int reg_num = fvt.out(op);  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num);
  }
  return result;
}

void exec_simt_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  m_warp_state.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp_state[k] = new WarpState(512, 512, m_config->warp_size);
    m_warp[k] = new warp_exec_t(this, m_config->warp_size);
  }
}

void simt_core_ctx::create_front_pipeline() {
  // pipeline_stages is the sum of normal pipeline stages and specialized_unit
  // stages * 2 (for ID and EX)
  unsigned total_pipeline_stages =
      N_PIPELINE_STAGES + m_config->m_specialized_unit.size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages);
  for (int j = 0; j < N_PIPELINE_STAGES; j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->pipe_widths[j], pipeline_stage_name_decode[j]));
  }
  for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1;
    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]);
  }
  for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1;
  }

  if (m_config->sub_core_model) {
    // in subcore model, each scheduler should has its own issue register, so
    // ensure num scheduler = reg width
    assert(m_config->opu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SP].get_size());
    assert(m_config->opu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SFU].get_size());
    assert(m_config->opu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_MEM].get_size());
    if (m_config->opu_tensor_core_avail)
      assert(m_config->opu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_TENSOR_CORE].get_size());
    if (m_config->opu_num_dp_units > 0)
      assert(m_config->opu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_DP].get_size());
    if (m_config->opu_num_int_units > 0)
      assert(m_config->opu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_INT].get_size());
    for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
      if (m_config->m_specialized_unit[j].num_units > 0)
        assert(m_config->opu_num_sched_per_core ==
               m_config->m_specialized_unit[j].id_oc_spec_reg_width);
    }
  }

  m_not_completed = 0;
  m_active_threads.reset();
  m_n_active_cta = 0;
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;

#if 0
  FIXME
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }
#endif

#if 0
  // m_icnt = new shader_memory_interface(this,cluster);
  if (m_config->opu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator =
      new simt_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);
#endif
  // fetch
  m_last_warp_fetched = 0;

#define STRSIZE 1024
  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sid);
  m_L0C = new l0_ccache(this, name, m_config->m_L0C_config,m_sid,INSTRUCTION,/*m_icnt*/nullptr,IN_L1I_MISS_QUEUE);
  /*
  m_L0C = new read_only_cache(name, m_config->m_L0C_config, m_sid,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE);
                              */
}

void simt_core_ctx::create_schedulers() {
  m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);

  // scedulers
  // must currently occur after all inputs have been initialized.
  std::string sched_config = m_config->opu_scheduler_string;
  const concrete_scheduler scheduler =
      sched_config.find("lrr") != std::string::npos
          ? CONCRETE_SCHEDULER_LRR
          : sched_config.find("two_level_active") != std::string::npos
                ? CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE
                : sched_config.find("gto") != std::string::npos
                      ? CONCRETE_SCHEDULER_GTO
                      : sched_config.find("rrr") != std::string::npos
                            ? CONCRETE_SCHEDULER_RRR
                      : sched_config.find("old") != std::string::npos
                            ? CONCRETE_SCHEDULER_OLDEST_FIRST
                            : sched_config.find("warp_limiting") !=
                                      std::string::npos
                                  ? CONCRETE_SCHEDULER_WARP_LIMITING
                                  : NUM_CONCRETE_SCHEDULERS;
  assert(scheduler != NUM_CONCRETE_SCHEDULERS);

  for (unsigned i = 0; i < m_config->opu_num_sched_per_core; i++) {
    switch (scheduler) {
      case CONCRETE_SCHEDULER_LRR:
        schedulers.push_back(new lrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE:
        schedulers.push_back(new two_level_active_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->opu_scheduler_string));
        break;
      case CONCRETE_SCHEDULER_GTO:
        schedulers.push_back(new gto_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_RRR:
        schedulers.push_back(new rrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_OLDEST_FIRST:
        schedulers.push_back(new oldest_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_WARP_LIMITING:
        schedulers.push_back(new swl_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->opu_scheduler_string));
        break;
      default:
        abort();
    };
  }

  for (unsigned i = 0; i < m_warp.size(); i++) {
    // distribute i's evenly though schedulers;
    schedulers[i % m_config->opu_num_sched_per_core]->add_supervised_warp_id(
        i);
  }
  for (unsigned i = 0; i < m_config->opu_num_sched_per_core; ++i) {
    schedulers[i]->done_adding_supervised_warps();
  }
}

void simt_core_ctx::create_exec_pipeline() {
  // op collector configuration
  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };

  opndcoll_rfu_t::port_vector_t in_ports;
  opndcoll_rfu_t::port_vector_t out_ports;
  opndcoll_rfu_t::uint_vector_t cu_sets;

  // configure generic collectors
  m_operand_collector.add_cu_set(
      GEN_CUS, m_config->opu_operand_collector_num_units_gen,
      m_config->opu_operand_collector_num_out_ports_gen);

  for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_gen;
       i++) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
    if (m_config->opu_tensor_core_avail) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
    }
    if (m_config->opu_num_dp_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
    }
    if (m_config->opu_num_int_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
    }
    if (m_config->m_specialized_unit.size() > 0) {
      for (unsigned j = 0; j < m_config->m_specialized_unit.size(); ++j) {
        in_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].ID_OC_SPEC_ID]);
        out_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].OC_EX_SPEC_ID]);
      }
    }
    cu_sets.push_back((unsigned)GEN_CUS);
    m_operand_collector.add_port(in_ports, out_ports, cu_sets);
    in_ports.clear(), out_ports.clear(), cu_sets.clear();
  }

  if (m_config->enable_specialized_operand_collector) {
    m_operand_collector.add_cu_set(
        SP_CUS, m_config->opu_operand_collector_num_units_sp,
        m_config->opu_operand_collector_num_out_ports_sp);
    m_operand_collector.add_cu_set(
        DP_CUS, m_config->opu_operand_collector_num_units_dp,
        m_config->opu_operand_collector_num_out_ports_dp);
    m_operand_collector.add_cu_set(
        TENSOR_CORE_CUS,
        m_config->opu_operand_collector_num_units_tensor_core,
        m_config->opu_operand_collector_num_out_ports_tensor_core);
    m_operand_collector.add_cu_set(
        SFU_CUS, m_config->opu_operand_collector_num_units_sfu,
        m_config->opu_operand_collector_num_out_ports_sfu);
    m_operand_collector.add_cu_set(
        MEM_CUS, m_config->opu_operand_collector_num_units_mem,
        m_config->opu_operand_collector_num_out_ports_mem);
    m_operand_collector.add_cu_set(
        INT_CUS, m_config->opu_operand_collector_num_units_int,
        m_config->opu_operand_collector_num_out_ports_int);

    for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_sp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_dp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
      cu_sets.push_back((unsigned)DP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_sfu;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
      cu_sets.push_back((unsigned)SFU_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0;
         i < m_config->opu_operand_collector_num_in_ports_tensor_core; i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
      cu_sets.push_back((unsigned)TENSOR_CORE_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_mem;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
      cu_sets.push_back((unsigned)MEM_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->opu_operand_collector_num_in_ports_int;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
      cu_sets.push_back((unsigned)INT_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
  }

  m_operand_collector.init(m_config->opu_num_reg_banks, this);

  m_num_function_units =
      m_config->opu_num_sp_units + m_config->opu_num_dp_units +
      m_config->opu_num_sfu_units + m_config->opu_num_tensor_core_units +
      m_config->opu_num_int_units + m_config->m_specialized_unit_num +
      1;  // sp_unit, sfu, dp, tensor, int, ldst_unit
  // m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
  // m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];

  // m_fu = new simd_function_unit*[m_num_function_units];

  for (unsigned k = 0; k < m_config->opu_num_sp_units; k++) {
    m_fu.push_back(new sp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SP);
    m_issue_port.push_back(OC_EX_SP);
  }

  for (unsigned k = 0; k < m_config->opu_num_dp_units; k++) {
    m_fu.push_back(new dp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_DP);
    m_issue_port.push_back(OC_EX_DP);
  }
  for (unsigned k = 0; k < m_config->opu_num_int_units; k++) {
    m_fu.push_back(new int_unit(&m_pipeline_reg[EX_WB], m_config->max_int_latency, this, k));
    m_dispatch_port.push_back(ID_OC_INT);
    m_issue_port.push_back(OC_EX_INT);
  }

  for (unsigned k = 0; k < m_config->opu_num_sfu_units; k++) {
    m_fu.push_back(new sf_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SFU);
    m_issue_port.push_back(OC_EX_SFU);
  }

  for (unsigned k = 0; k < m_config->opu_num_tensor_core_units; k++) {
    m_fu.push_back(new tensor_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_TENSOR_CORE);
    m_issue_port.push_back(OC_EX_TENSOR_CORE);
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    for (unsigned k = 0; k < m_config->m_specialized_unit[j].num_units; k++) {
      m_fu.push_back(new specialized_unit(
          &m_pipeline_reg[EX_WB], m_config, this, SPEC_UNIT_START_ID + j,
          m_config->m_specialized_unit[j].name,
          m_config->m_specialized_unit[j].latency, k));
      m_dispatch_port.push_back(m_config->m_specialized_unit[j].ID_OC_SPEC_ID);
      m_issue_port.push_back(m_config->m_specialized_unit[j].OC_EX_SPEC_ID);
    }
  }
#if 1
  m_ldst_unit = new ldst_unit(/*m_icnt, m_mem_fetch_allocator,*/ this,
                              &m_operand_collector, m_scoreboard, m_config->smem_latency,
                              /*m_memory_config, */m_stats, m_sid, m_tpc);
  m_fu.push_back(m_ldst_unit);
#endif
  m_dispatch_port.push_back(ID_OC_MEM);
  m_issue_port.push_back(OC_EX_MEM);

  assert(m_num_function_units == m_fu.size() and
         m_fu.size() == m_dispatch_port.size() and
         m_fu.size() == m_issue_port.size());

  // there are as many result buses as the width of the EX_WB stage
  num_result_bus = m_config->pipe_widths[EX_WB];
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }
}

simt_core_ctx::simt_core_ctx(class opu_sim *opu,
                                 class simt_core_cluster *cluster,
                                 unsigned shader_id, unsigned tpc_id,
                                 const simtcore_config *config,
                                 /*const memory_config *mem_config,*/
                                 simt_core_stats *stats)
    : core_t(opu, config->warp_size, config->n_thread_per_shader),
      m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size),
      m_active_warps(0),
      m_dynamic_warp_id(0) {
    // TODO schi 
    m_kernel_finishing = false;

  m_cluster = cluster;
  m_config = config;
  // m_memory_config = mem_config;
  m_stats = stats;
  unsigned warp_size = config->warp_size;
  Issue_Prio = 0;

  m_sid = shader_id;
  m_tpc = tpc_id;

  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;

  // Jin: for concurrent kernels on a SM
  m_occupied_n_threads = 0;
  m_occupied_shmem = 0;
  m_occupied_regs = 0;
  m_occupied_ctas = 0;
  m_occupied_hwtid.reset();
  m_occupied_cta_to_hwtid.clear();
}

void simt_core_ctx::reinit(unsigned start_thread, unsigned end_thread,
                             bool reset_not_completed) {
  if (reset_not_completed) {
    m_not_completed = 0;
    m_active_threads.reset();

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();
    m_active_warps = 0;
  }
  /*
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  */
  for (unsigned i = start_thread / m_config->warp_size;
       i < end_thread / m_config->warp_size; ++i) {
    m_warp[i]->reset();
    m_simt_stack[i]->reset();
  }
}

void simt_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, KernelInfo &kernel) {
  //
  address_type start_pc = 0; // next_pc(start_thread);
  // unsigned kernel_id = kernel.get_uid();
  //if (m_config->model == POST_DOMINATOR) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);
    for (unsigned i = start_warp; i < end_warp; ++i) {
      unsigned n_active = 0;
      simt_mask_t active_threads;
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          m_active_threads.set(hwtid);
          active_threads.set(t);
        }
      }
      m_simt_stack[i]->launch(start_pc, active_threads);

      m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id);
      ++m_dynamic_warp_id;
      m_not_completed += n_active;
      ++m_active_warps;
    }
  //}
}

// return the next pc of a thread
#if 0
address_type simt_core_ctx::next_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
}
#endif


void simt_core_ctx::get_pdom_stack_top_info(unsigned tid, unsigned *pc,
                                              unsigned *rpc) const {
  unsigned warp_id = tid / m_config->warp_size;
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

float simt_core_ctx::get_current_occupancy(unsigned long long &active,
                                             unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) {
    total += m_warp.size();
    active += m_active_warps;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

// extern long long g_program_memory_start;
//#define PROGRAM_MEM_START g_program_memory_start;

#if 0
#define PROGRAM_MEM_START                                      \
  0xF0000000 /* should be distinct from other memory spaces... \
                check ptx_ir.h to verify this does not overlap \
                other memory spaces */

#endif

const warp_inst_t *exec_simt_core_ctx::get_next_inst(unsigned warp_id,
                                                       address_type pc) {
  // read the inst from the functional model
  // return m_opuusim->opu_ctx->ptx_fetch_inst(pc);
  // FIXME
  return nullptr;
}

void exec_simt_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                   const warp_inst_t *pI,
                                                   unsigned *pc,
                                                   unsigned *rpc) {
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

const active_mask_t &exec_simt_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  return m_simt_stack[warp_id]->get_active_mask();
}

void simt_core_ctx::decode() {
  if (m_inst_fetch_buffer.m_valid) {
    // decode 1 or 2 instructions and place them into ibuffer
    address_type pc = m_inst_fetch_buffer.m_pc;
    const warp_inst_t *pI1 = get_next_inst(m_inst_fetch_buffer.m_warp_id, pc);
    m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(0, pI1);
    m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
    if (pI1) {
      m_stats->m_num_decoded_insn[m_sid]++;
      const warp_inst_t *pI2 =
          get_next_inst(m_inst_fetch_buffer.m_warp_id, pc + pI1->GetSize());
      if (pI2) {
        m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(1, pI2);
        m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
        m_stats->m_num_decoded_insn[m_sid]++;
      }
    }
    m_inst_fetch_buffer.m_valid = false;
  }
}

void simt_core_ctx::fetch() {
  if (!m_inst_fetch_buffer.m_valid) {
    if (m_L0C->access_ready()) {
      mem_fetch *mf = m_L0C->next_access();
      m_warp[mf->get_wid()]->clear_imiss_pending();
      m_inst_fetch_buffer =
          ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
                          mf->get_access_size(), mf->get_wid());
      /*TODO schi
          assert(m_warp[mf->get_wid()]->get_pc() ==
             (mf->get_addr() -
              PROGRAM_MEM_START));  // Verify that we got the instruction we
                                    // were expecting.
            */

      m_inst_fetch_buffer.m_valid = true;
      m_warp[mf->get_wid()]->set_last_fetch(m_opuusim->gpu_sim_cycle);
      delete mf;
    } else {
      // find an active warp with space in instruction buffer that is not
      // already waiting on a cache miss and get next 1-2 instructions from
      // i-cache...
      for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        unsigned warp_id =
            (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;

        // this code checks if this warp has finished executing and can be
        // reclaimed
#if 0
        if (m_warp[warp_id]->hardware_done() &&
            !m_scoreboard->pendingWrites(warp_id) &&
            !m_warp[warp_id]->done_exit()) {
          bool did_exit = false;
          for (unsigned t = 0; t < m_config->warp_size; t++) {
            unsigned tid = warp_id * m_config->warp_size + t;
            if (m_threadState[tid].m_active == true) {
              m_threadState[tid].m_active = false;
              unsigned cta_id = m_warp[warp_id]->get_cta_id();
              if (m_thread[tid] == NULL) {
                register_cta_thread_exit(cta_id, m_kernel);
              } else {
                register_cta_thread_exit(cta_id,
                                         &(m_thread[tid]->get_kernel()));
              }
              m_not_completed -= 1;
              m_active_threads.reset(tid);
              did_exit = true;
            }
          }
          if (did_exit) m_warp[warp_id]->set_done_exit();
          --m_active_warps;
          assert(m_active_warps >= 0);
        }
#endif
        // this code fetches instructions from the i-cache or generates memory
        if (!m_warp[warp_id]->functional_done() &&
            !m_warp[warp_id]->imiss_pending() &&
            m_warp[warp_id]->ibuffer_empty()) {
          address_type pc;
          pc = m_warp[warp_id]->get_pc();
          // FIXME FIXME
          // address_type ppc = pc + PROGRAM_MEM_START;
          address_type ppc = pc + m_kernel->get_inst_base_vaddr();
          unsigned nbytes = 16;
          unsigned offset_in_block =
              pc & (m_config->m_L0C_config.get_line_sz() - 1);
          if ((offset_in_block + nbytes) > m_config->m_L0C_config.get_line_sz())
            nbytes = (m_config->m_L0C_config.get_line_sz() - offset_in_block);

          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          // HACK: This access will get sent into gem5, so the control
          // header size must be zero, since gem5 packets will assess
          // control header sizing
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_opuusim->opu_ctx);
          mem_fetch *mf = new mem_fetch(
              acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
              warp_id, m_sid, m_tpc,
              m_opuusim->gpu_tot_sim_cycle + m_opuusim->gpu_sim_cycle);
          std::list<cache_event> events;
          enum cache_request_status status;
          if (m_config->perfect_inst_const_cache){
            status = HIT;
            // shader_cache_access_log(m_sid, INSTRUCTION, 0);
          }
          else
            status = m_L0C->access(
                (address_type)ppc, mf,
                m_opuusim->gpu_sim_cycle + m_opuusim->gpu_tot_sim_cycle, events);

          if (status == MISS) {
            m_last_warp_fetched = warp_id;
            m_warp[warp_id]->set_imiss_pending();
            m_warp[warp_id]->set_last_fetch(m_opuusim->gpu_sim_cycle);
          } else if (status == HIT) {
            m_last_warp_fetched = warp_id;
            m_inst_fetch_buffer = ifetch_buffer_t(pc, nbytes, warp_id);
            m_warp[warp_id]->set_last_fetch(m_opuusim->gpu_sim_cycle);
            delete mf;
          } else {
            m_last_warp_fetched = warp_id;
            assert(status == RESERVATION_FAIL);
            delete mf;
          }
          break;
        }
      }
    }
  }

  m_L0C->cycle();
}

void exec_simt_core_ctx::func_exec_inst(warp_inst_t &inst) {
  execute_warp_inst_t(inst);
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
    // inst.print_m_accessq();
  }
}

// TODO schi add
void simt_core_ctx::warp_reaches_barrier(OpuWarpinst &inst_) {
    warp_inst_t &inst = dynamic_cast<warp_inst_t&>(inst_);
    m_barriers.warp_reaches_barrier(m_warp[inst.warp_id()]->get_cta_id(), inst.warp_id(), &inst);
}

void simt_core_ctx::issue_warp(register_set &pipe_reg_set,
                                 const warp_inst_t *next_inst,
                                 const active_mask_t &active_mask,
                                 unsigned warp_id, unsigned sch_id) {
  warp_inst_t **pipe_reg =
      pipe_reg_set.get_free(m_config->sub_core_model, sch_id);
  assert(pipe_reg);

  m_warp[warp_id]->ibuffer_free();
  assert(next_inst->valid());
  **pipe_reg = *next_inst;  // static instruction information
  (*pipe_reg)->issue(active_mask, warp_id,
                     m_opuusim->gpu_tot_sim_cycle + m_opuusim->gpu_sim_cycle,
                     m_warp[warp_id]->get_dynamic_warp_id(),
                     sch_id);  // dynamic instruction information
  m_stats->shader_cycle_distro[2 + (*pipe_reg)->active_count()]++;
  func_exec_inst(**pipe_reg);

  if (next_inst->op == opu_op_t::BARRIER_OP) {
    m_warp[warp_id]->store_info_of_last_inst_at_barrier(*pipe_reg);
    m_barriers.warp_reaches_barrier(m_warp[warp_id]->get_cta_id(), warp_id,
                                    const_cast<warp_inst_t *>(next_inst));

  } else if (next_inst->op == opu_op_t::MEMORY_BARRIER_OP) {
    m_warp[warp_id]->set_membar();
  }

  updateSIMTStack(warp_id, *pipe_reg);

  m_scoreboard->reserveRegisters(*pipe_reg);
  m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->GetSize());
}

void simt_core_ctx::issue() {
  // Ensure fair round robin issu between schedulers
  unsigned j;
  for (unsigned i = 0; i < schedulers.size(); i++) {
    j = (Issue_Prio + i) % schedulers.size();
    schedulers[j]->cycle();
  }
  Issue_Prio = (Issue_Prio + 1) % schedulers.size();

  // really is issue;
  // for (unsigned i = 0; i < schedulers.size(); i++) {
  //    schedulers[i]->cycle();
  //}
}


void simt_core_ctx::read_operands() {
  for (int i = 0; i < m_config->reg_file_port_throughput; ++i)
    m_operand_collector.step();
}

address_type coalesced_segment(address_type addr,
                               unsigned segment_size_lg2bytes) {
  return (addr >> segment_size_lg2bytes);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned simt_core_ctx::translate_local_memaddr(
    address_type localaddr, unsigned tid, unsigned num_shader,
    unsigned datasize, address_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0;
  unsigned max_concurrent_threads = 0;
  if (m_config->opu_local_mem_map) {
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to contiguous
    // addresses, then distribute across memory space by CTAs from successive
    // shader cores first, then by successive CTA in same shader core
    thread_base =
        4 * (kernel_padded_threads_per_cta *
                 (m_sid + num_shader * (tid / kernel_padded_threads_per_cta)) +
             tid % kernel_padded_threads_per_cta);
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
  } else {
      assert("gem5-gpu does not work with legacy local mapping!\n");
    // legacy mapping that maps the same address in the local memory space of
    // all threads to a single contiguous address region
    thread_base = 4 * (m_config->n_thread_per_shader * m_sid + tid);
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4;
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(
        localaddr % 4 ==
        0);  // Address must be 4B aligned - required if accessing 4B per
             // request, otherwise access will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i;
      // TODO schi 
      // address_type linear_address = local_word * max_concurrent_threads * 4 +
      //                              thread_base + LOCAL_GENERIC_START;
      address_type linear_address = local_word*max_concurrent_threads*4 +
                              thread_base + m_gem5_getLocalBaseVaddr();
      assert(linear_address < m_gem5_getLocalBaseVaddr() + (LOCAL_MEM_SIZE_MAX * max_concurrent_threads));
      translated_addrs[i] = linear_address;
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4;
    address_type local_word_offset = localaddr % 4;
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B chunk
    // TODO schi
    // address_type linear_address = local_word * max_concurrent_threads * 4 +
    //                              local_word_offset + thread_base +
    //                              LOCAL_GENERIC_START;
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    local_word_offset + thread_base +
                                    m_gem5_getLocalBaseVaddr();
    assert(linear_address < m_gem5_getLocalBaseVaddr() + (LOCAL_MEM_SIZE_MAX * max_concurrent_threads));
    translated_addrs[0] = linear_address;
  }
  return num_accesses;
}

/////////////////////////////////////////////////////////////////////////////////////////
int simt_core_ctx::test_res_bus(int latency) {
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) {
      return i;
    }
  }
  return -1;
}

void simt_core_ctx::execute() {
  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1;
  }
  for (unsigned n = 0; n < m_num_function_units; n++) {
    unsigned multiplier = m_fu[n]->clock_multiplier();
    for (unsigned c = 0; c < multiplier; c++) m_fu[n]->cycle();
    m_fu[n]->active_lanes_in_pipeline();
    unsigned issue_port = m_issue_port[n];
    register_set &issue_inst = m_pipeline_reg[issue_port];
    unsigned reg_id;
    bool partition_issue =
        m_config->sub_core_model && m_fu[n]->is_issue_partitioned();
    if (partition_issue) {
      reg_id = m_fu[n]->get_issue_reg_id();
    }
    warp_inst_t **ready_reg = issue_inst.get_ready(partition_issue, reg_id);
    if (issue_inst.has_ready(partition_issue, reg_id) &&
        m_fu[n]->can_issue(**ready_reg)) {
      bool schedule_wb_now = !m_fu[n]->stallable();
      int resbus = -1;
      if (schedule_wb_now &&
          (resbus = test_res_bus((*ready_reg)->latency)) != -1) {
        assert((*ready_reg)->latency < MAX_ALU_LATENCY);
        m_result_bus[resbus]->set((*ready_reg)->latency);
        m_fu[n]->issue(issue_inst);
      } else if (!schedule_wb_now) {
        m_fu[n]->issue(issue_inst);
      } else {
        // stall issue (cannot reserve result bus)
      }
    }
  }
}


void simt_core_ctx::warp_inst_complete(const warp_inst_t &inst) {
#if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu \n",
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc,  m_opuusim->gpu_tot_sim_cycle +  m_opuusim->gpu_sim_cycle);

  if (inst.op_pipe == opu_pipeline_t::SP__OP)
    m_stats->m_num_sp_committed[m_sid]++;
  else if (inst.op_pipe == opu_pipeline_t::SFU__OP)
    m_stats->m_num_sfu_committed[m_sid]++;
  else if (inst.op_pipe == opu_pipeline_t::MEM__OP)
    m_stats->m_num_mem_committed[m_sid]++;

  if (m_config->opu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
  else
    m_stats->m_num_sim_insn[m_sid] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sid]++;
#endif
  m_opuusim->gpu_sim_insn += inst.active_count();
  inst.completed(m_opuusim->gpu_tot_sim_cycle + m_opuusim->gpu_sim_cycle);
}

void simt_core_ctx::writeback() {
  unsigned max_committed_thread_instructions =
      m_config->warp_size *
      (m_config->pipe_widths[EX_WB]);  // from the functional units
  m_stats->m_pipeline_duty_cycle[m_sid] =
      ((float)(m_stats->m_num_sim_insn[m_sid] -
               m_stats->m_last_num_sim_insn[m_sid])) /
      max_committed_thread_instructions;

  m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
  m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];

  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready();
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
     * The operand collector writeback can generally generate a stall
     * However, here, the pipelines should be un-stallable. This is
     * guaranteed because this is the first time the writeback function
     * is called after the operand collector's step function, which
     * resets the allocations. There is one case which could result in
     * the writeback function returning false (stall), which is when
     * an instruction tries to modify two registers (GPR and predicate)
     * To handle this case, we ignore the return value (thus allowing
     * no stalling).
     */

    m_operand_collector.writeback(*pipe_reg);
    unsigned warp_id = pipe_reg->warp_id();
    m_scoreboard->releaseRegisters(pipe_reg);
    m_warp[warp_id]->dec_inst_in_pipeline();
    warp_inst_complete(*pipe_reg);
    m_opuusim->gpu_sim_insn_last_update_sid = m_sid;
    m_opuusim->gpu_sim_insn_last_update = m_opuusim->gpu_sim_cycle;
    m_last_inst_gpu_sim_cycle = m_opuusim->gpu_sim_cycle;
    m_last_inst_gpu_tot_sim_cycle = m_opuusim->gpu_tot_sim_cycle;
    pipe_reg->clear();
    preg = m_pipeline_reg[EX_WB].get_ready();
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}


#if 0
void simt_core_ctx::register_cta_thread_exit(unsigned cta_num,
                                               KernelInfo *kernel) {
  assert(m_cta_status[cta_num] > 0);
  m_cta_status[cta_num]--;
  if (!m_cta_status[cta_num]) {
    // Increment the completed CTAs
    m_stats->ctas_completed++;
    m_opuusim->inc_completed_cta();
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    // shader_CTA_count_unlog(m_sid, 1);
//      printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld), %u CTAs running\n", m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle,
//             m_n_active_cta );
    // m_opuusim->gem5CudaGPU->getCudaCore(m_sid)->record_block_commit(cta_num);
    m_gem5_record_block_commit(cta_num);


    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_opuusim->gpu_sim_cycle, m_opuusim->gpu_tot_sim_cycle,
        m_n_active_cta);

    if (m_n_active_cta == 0) {
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_opuusim->kernel_more_cta_left(m_kernel));
      }
      m_kernel = NULL;
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel);
    kernel->dec_running();
    if (!m_opuusim->kernel_more_cta_left(kernel)) {
      if (!kernel->running()) {
        SHADER_DPRINTF(LIVENESS,
                       "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                       "finished on shader %u.\n",
                       kernel->get_uid(), kernel->name().c_str(), m_sid);

        if (m_kernel == kernel) m_kernel = NULL;
        m_opuusim->set_kernel_done(kernel);
      }
    }
  }
}

void simt_core_ctx::start_kernel_finish()
{
    assert(!m_kernel_finishing);
    m_kernel_finishing = true;
    m_opuusim->gem5CudaGPU->getCudaCore(m_sid)->finishKernel();
}
#endif

void simt_core_ctx::finish_kernel()
{
  assert( m_kernel != NULL );
  m_kernel->dec_running();
  printf("GPGPU-Sim uArch: Shader %u empty (release kernel %u ).\n", m_sid, m_kernel->get_uid()
         /*m_kernel->name().c_str()*/ );
  if ( m_kernel->no_more_ctas_to_run() ) {
      if ( !m_kernel->running() ) {
          printf("GPGPU-Sim uArch: GPU detected kernel finished on shader %u.\n"/*, m_kernel->name().c_str()*/, m_sid );
          m_opuusim->set_kernel_done( m_kernel );
      }
  }
  m_kernel=NULL;
  m_kernel_finishing = false;
  fflush(stdout);
}

#if 0
void simt_core_ctx::incexecstat(warp_inst_t *&inst)
{
  if(inst->mem_op==TEX)
    inctex_stat(inst->active_count(),1);
}
#endif

void simt_core_ctx::print_stage(unsigned int stage, FILE *fout) const {
  // m_pipeline_reg[stage].print(fout);
}

void simt_core_ctx::display_simt_state(FILE *fout, int mask) const {
    /*
  if ((mask & 4) && m_config->model == POST_DOMINATOR) {
    fprintf(fout, "per warp SIMT control-flow state:\n");
    unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
    for (unsigned i = 0; i < n; i++) {
      unsigned nactive = 0;
      for (unsigned j = 0; j < m_config->warp_size; j++) {
        unsigned tid = i * m_config->warp_size + j;
        int done = ptx_thread_done(tid);
        nactive += (ptx_thread_done(tid) ? 0 : 1);
        if (done && (mask & 8)) {
          unsigned done_cycle = m_thread[tid]->donecycle();
          if (done_cycle) {
            printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle);
          }
        }
      }
      if (nactive == 0) {
        continue;
      }
      m_simt_stack[i]->print(fout);
    }
    fprintf(fout, "\n");
  }
  */
}


void simt_core_ctx::display_pipeline(FILE *fout, int print_mem,
                                       int mask) const {
  fprintf(fout, "=================================================\n");
  fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid,
          m_opuusim->gpu_tot_sim_cycle, m_opuusim->gpu_sim_cycle, m_not_completed);
  fprintf(fout, "=================================================\n");

  // dump_warp_state(fout);
  fprintf(fout, "\n");

  m_L0C->display_state(fout);

  fprintf(fout, "IF/ID       = ");
  if (!m_inst_fetch_buffer.m_valid)
    fprintf(fout, "bubble\n");
  else {
    fprintf(fout, "w%2u : pc = 0x%x, nbytes = %u\n",
            m_inst_fetch_buffer.m_warp_id, m_inst_fetch_buffer.m_pc,
            m_inst_fetch_buffer.m_nbytes);
  }
  fprintf(fout, "\nibuffer status:\n");
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (!m_warp[i]->ibuffer_empty()) m_warp[i]->print_ibuffer(fout);
  }
  fprintf(fout, "\n");
  display_simt_state(fout, mask);
  fprintf(fout, "-------------------------- Scoreboard\n");
  m_scoreboard->printContents();
  /*
     fprintf(fout,"ID/OC (SP)  = ");
     print_stage(ID_OC_SP, fout);
     fprintf(fout,"ID/OC (SFU) = ");
     print_stage(ID_OC_SFU, fout);
     fprintf(fout,"ID/OC (MEM) = ");
     print_stage(ID_OC_MEM, fout);
  */
  fprintf(fout, "-------------------------- OP COL\n");
  m_operand_collector.dump(fout);
  /* fprintf(fout, "OC/EX (SP)  = ");
     print_stage(OC_EX_SP, fout);
     fprintf(fout, "OC/EX (SFU) = ");
     print_stage(OC_EX_SFU, fout);
     fprintf(fout, "OC/EX (MEM) = ");
     print_stage(OC_EX_MEM, fout);
  */
  fprintf(fout, "-------------------------- Pipe Regs\n");

  for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
    fprintf(fout, "--- %s ---\n", pipeline_stage_name_decode[i]);
    print_stage(i, fout);
    fprintf(fout, "\n");
  }

  fprintf(fout, "-------------------------- Fu\n");
  for (unsigned n = 0; n < m_num_function_units; n++) {
    // m_fu[n]->print(fout);
    fprintf(fout, "---------------\n");
  }
  fprintf(fout, "-------------------------- other:\n");

  for (unsigned i = 0; i < num_result_bus; i++) {
    std::string bits = m_result_bus[i]->to_string();
    fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str());
  }
  fprintf(fout, "EX/WB      = ");
  print_stage(EX_WB, fout);
  fprintf(fout, "\n");
  fprintf(
      fout,
      "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);

  if (m_active_threads.count() <= 2 * m_config->warp_size) {
    fprintf(fout, "Active Threads : ");
    unsigned last_warp_id = -1;
    for (unsigned tid = 0; tid < m_active_threads.size(); tid++) {
      unsigned warp_id = tid / m_config->warp_size;
      if (m_active_threads.test(tid)) {
        if (warp_id != last_warp_id) {
          fprintf(fout, "\n  warp %u : ", warp_id);
          last_warp_id = warp_id;
        }
        fprintf(fout, "%u ", tid);
      }
    }
  }
}


void simt_core_ctx::cycle() {
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sid]++;
  writeback();
  execute();
  read_operands();
  issue();
  for (int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    decode();
    fetch();
  }
}

// Flushes all content of the cache to memory

void simt_core_ctx::cache_flush() { m_ldst_unit->flush(); }

void simt_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }


warp_exec_t* simt_core_ctx::get_warp(unsigned warp_id) {
    return m_warp[warp_id];
}

WarpState* simt_core_ctx::get_warp_state(unsigned warp_id) const {
    return m_warp_state[warp_id];
}

#if 0
void simt_core_ctx::warp_exit(unsigned warp_id) {
  bool done = true;
  for (unsigned i = warp_id * get_config()->warp_size;
       i < (warp_id + 1) * get_config()->warp_size; i++) {
    //		if(this->m_thread[i]->m_functional_model_thread_state &&
    // this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
    // done = false;
    //		}

    if (m_thread[i] && !m_thread[i]->is_done()) done = false;
  }
  // if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
  // if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
  if (done) m_barriers.warp_exit(warp_id);
}
#endif

active_mask_t simt_core_ctx::warp_active_mask(uint32_t warp_id) {
    return m_warp[warp_id]->m_active_threads;
}

bool simt_core_ctx::check_if_non_released_reduction_barrier(
    warp_inst_t &inst) {
  unsigned warp_id = inst.warp_id();
  bool bar_red_op = (inst.op == opu_op_t::BARRIER_OP) && (inst.bar_type == barrier_t::RED);
  bool non_released_barrier_reduction = false;
  bool warp_stucked_at_barrier = warp_waiting_at_barrier(warp_id);
  bool single_inst_in_pipeline =
      (m_warp[warp_id]->num_issued_inst_in_pipeline() == 1);
  non_released_barrier_reduction =
      single_inst_in_pipeline and warp_stucked_at_barrier and bar_red_op;
  printf("non_released_barrier_reduction=%u\n", non_released_barrier_reduction);
  return non_released_barrier_reduction;
}

bool simt_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool simt_core_ctx::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_warp[warp_id]->get_membar()) return false;
  if (!m_scoreboard->pendingWrites(warp_id)) {
    m_warp[warp_id]->clear_membar();
#if 0
    // FIXME
    if (m_opuusim->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
      cache_invalidate();
      // TO DO: you need to stall the SM for 5k cycles.
    }
#endif
    return false;
  }
  return true;
}

#if 1
void simt_core_ctx::set_max_cta(const KernelInfo &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size;
}
#endif

void simt_core_ctx::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_warp[wid]->get_n_atomic() >= n);
  m_warp[wid]->dec_n_atomic(n);
}

void simt_core_ctx::broadcast_barrier_reduction(unsigned cta_id,
                                                  unsigned bar_id,
                                                  warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      const warp_inst_t *inst =
          m_warp[i]->restore_info_of_last_inst_at_barrier();
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask());
    }
  }
}

bool simt_core_ctx::fetch_unit_response_buffer_full() const { return false; }

void simt_core_ctx::accept_fetch_response(gem5::OpuMemfetch *mf) {
#if 0
  mf->set_status(IN_SHADER_FETCHED,
                 m_opuusim->gpu_sim_cycle + m_opuusim->gpu_tot_sim_cycle);
  m_L0C->fill(mf, m_opuusim->gpu_sim_cycle + m_opuusim->gpu_tot_sim_cycle);
#endif
}

bool simt_core_ctx::ldst_unit_response_buffer_full() const {
  return m_ldst_unit->response_buffer_full();
}

#if 0
void simt_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit->fill(mf);
}

void simt_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->opu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  m_warp[warp_id]->dec_store_req();
}
#endif

#if 0
void simt_core_ctx::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                        unsigned &dl1_misses) {
  m_ldst_unit->print_cache_stats(fp, dl1_accesses, dl1_misses);
}

void simt_core_ctx::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L0C->get_stats();          // Get L1I stats
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void simt_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L0C) m_L0C->get_sub_stats(css);
}
void simt_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css);
}
void simt_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1C_sub_stats(css);
}
void simt_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1T_sub_stats(css);
}

void simt_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
                                           long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sid];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sid];
}
#endif

void exec_simt_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                         unsigned t,
                                                         unsigned tid) {
  uint32_t wid = inst.warp_id();
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();
  if (inst.get_space() == opu_mspace_t::PRIVATE && (inst.is_load() || inst.is_store())) {
    address_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (address_type *)localaddrs);
    inst.set_addr(t, (address_type *)localaddrs, num_addrs);
  }
  //if (ptx_thread_done(tid)) {
  if (get_warp_state(wid)->isLaneExit(tid)) {
    m_warp[inst.warp_id()]->set_completed(t);
    m_warp[inst.warp_id()]->ibuffer_flush();
  }

  // PC-Histogram Update
  unsigned warp_id = inst.warp_id();
  unsigned pc = inst.pc;
  for (unsigned t = 0; t < m_config->warp_size; t++) {
    if (inst.active(t)) {
      int tid = warp_id * m_config->warp_size + t;
      // cflog_update_thread_pc(m_sid, tid, pc);
    }
  }
}

bool simt_core_ctx::fence_unblock_needed(unsigned warp_id) {
    return m_warp[warp_id]->get_membar();
}
void simt_core_ctx::complete_fence(unsigned warp_id) {
    assert(m_warp[warp_id]->get_membar());
    m_warp[warp_id]->clear_membar();
}

bool simt_core_ctx::ldst_unit_wb_inst(OpuWarpinst &inst) {
    return m_ldst_unit->writebackInst(dynamic_cast<warp_inst_t&>(inst));
}

void simt_core_ctx::inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }

void simt_core_ctx::dec_inst_in_pipeline(unsigned warp_id) {
  m_warp[warp_id]->dec_inst_in_pipeline();
}  // also used in writeback()

void simt_core_ctx::set_kernel(KernelInfo *k) {
    assert(k);
    m_kernel = k;
    k->inc_running();
    // printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
    //       m_kernel->get_uid(), m_kernel->name().c_str());
}
void simt_core_ctx::issue_block2core(KernelInfo &kernel) {
  if (!m_config->opu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->opu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->opu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  unsigned ctaid = kernel.get_next_cta_id_single();
#if 0
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;
#endif

  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  //shader_CTA_count_log(m_sid, 1);
  //SHADER_DPRINTF(LIVENESS,
  //               "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
  //               "initialized @(%lld,%lld)\n",
  //               free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
  //               m_gpu->gpu_tot_sim_cycle);
  // m_gpu->gem5OpuTop->getCudaCore(m_sid)->record_block_issue(free_cta_hw_id);
}

#if 1
bool simt_core_ctx::can_issue_1block(KernelInfo &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->opu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int simt_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool simt_core_ctx::occupy_shader_resource_1block(KernelInfo &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  // const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  // const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + k.get_shared_memsize() > m_config->opu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((k.get_vreg_used() + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->opu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += k.get_shared_memsize();
    m_occupied_regs += (padded_cta_size * ((k.get_vreg_used() + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas);
  }

  return true;
}

void simt_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     KernelInfo &k) {
  if (m_config->opu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    // const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    // const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)k.get_shared_memsize());
    m_occupied_shmem -= k.get_shared_memsize();

    unsigned int used_regs = padded_cta_size * ((k.get_vreg_used() + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}
#endif

