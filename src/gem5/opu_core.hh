/*
 * Copyright (c) 2011 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __OPU_CORE_HH__
#define __OPU_CORE_HH__

#include <map>
#include <queue>
#include <set>

#include "cpu/translation.hh"
// #include "cuda-sim/ptx.tab.h"
// #include "cuda-sim/ptx_ir.h"
// #include "cuda-sim/ptx_sim.h"
// #include "gpgpu-sim/OpuMemfetch.h"
//#include "../core/shader.h"
#include "coasm.h"
#include "../opucore_base.h"
#include "../opuinst_base.h"
#include "gpu/atomic_operations.hh"
#include "gpu/shader_tlb.hh"
#include "sim/clocked_object.hh"
#include "mem/ruby/system/RubySystem.hh"
#include "params/OpuCore.hh"

namespace gem5 {

class OpuTop;


/**
 *  Wrapper class for the shader cores in GPGPU-Sim. Shader memory references
 *  to GPU const, global and instruction memory go through this wrapper to the
 *  gem5-side memory hierarchy.
 *
 *  NOTE: A CUDA core is equivalent to an NVIDIA streaming multiprocessor (SM)
 */
class OpuCore : public ClockedObject
{
  protected:
    typedef OpuCoreParams Params;
    // Functions to translate GPGPU-Sim values to gem5-gpu values
    static AtomicOpRequest::Operation
    getAtomOpType(opu_atomic_t opu_sim_value) {
        switch (opu_sim_value) {
            case opu_atomic_t::ATOMIC_CAS:
            return AtomicOpRequest::ATOMIC_CAS_OP;
            case opu_atomic_t::ATOMIC_ADD:
            return AtomicOpRequest::ATOMIC_ADD_OP;
            case opu_atomic_t::ATOMIC_INC:
            return AtomicOpRequest::ATOMIC_INC_OP;
            case opu_atomic_t::ATOMIC_MIN:
            return AtomicOpRequest::ATOMIC_MIN_OP;
            case opu_atomic_t::ATOMIC_MAX:
            return AtomicOpRequest::ATOMIC_MAX_OP;
          default:
            panic("Unknown atomic type: %u\n", static_cast<uint32_t>(opu_sim_value));
            break;
        }
        return AtomicOpRequest::ATOMIC_INVALID_OP;
    }

    static AtomicOpRequest::DataType
    getDataType(opu_datatype_t opu_sim_value) {
        switch (opu_sim_value) {
            case opu_datatype_t::DT_INT32:
            return AtomicOpRequest::S32_TYPE_;
            case opu_datatype_t::DT_UINT32:
            return AtomicOpRequest::U32_TYPE_;
            case opu_datatype_t::DT_FP32:
            return AtomicOpRequest::F32_TYPE_;
            case opu_datatype_t::DT_B32:
            return AtomicOpRequest::B32_TYPE_;
          default:
            panic("Unknown atomic data type: %u\n", static_cast<uint32_t>(opu_sim_value));
            break;
        }
        return AtomicOpRequest::INVALID_TYPE;
    }
    /**
     * Port for sending a receiving instruction memory accesses
     * Required for implementing ClockedObject
     */
    class InstPort : public MasterPort
    {
        friend class OpuCore;

      private:
        // Pointer back to shader core for callbacks
        OpuCore *core;

      public:
        InstPort(const std::string &_name, OpuCore *_core)
        : MasterPort(_name, _core), core(_core) {}

      protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvReqRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
    };
    // Instantiation of above port
    InstPort instPort;

    /**
     * Port to send packets to the load/store queue and coalescer
     */
    class LSQPort : public MasterPort
    {
        friend class OpuCore;

      private:
        OpuCore *core;
        int idx;

      public:
        LSQPort(const std::string &_name, OpuCore *_core, int _idx)
        : MasterPort(_name, _core), core(_core), idx(_idx) {}

      protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvReqRetry();
    };
    // Ports for each of the GPU lanes
    std::vector<LSQPort*> lsqPorts;

    /**
     * A port to send control commands to the LSQ. Currently, this is used
     * to send the flush command on kernel boundaries. Functions more like a
     * register for transmitting control logic than a true port, so receive
     * functions do not block.
     */
    class LSQControlPort : public MasterPort
    {
        friend class OpuCore;

      private:
        OpuCore *core;

      public:
        LSQControlPort(const std::string &_name, OpuCore *_core)
        : MasterPort(_name, _core), core(_core) {}

      protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvReqRetry();
    };
    LSQControlPort lsqControlPort;

    // Port that is blocked. If -1 then no port is blocked.
    int writebackBlocked;

    class SenderState : public Packet::SenderState {
    public:
        SenderState(const OpuWarpinst &_inst) : inst(_inst) {}
        const OpuWarpinst &inst;
    };

    const Params * params() const {
        return dynamic_cast<const Params *>(_params);
    }
    const OpuCoreParams *_params;

    RequestorID dataMasterId;
    RequestorID instMasterId;

  private:

    // ID for this CUDA core, should match the id in GPGPU-Sim
    int id;

    // Number of threads in the warp, also the number of lanes per SM
    int warpSize;

    // Stalled because a memory request called recvReqRetry, usually because
    // a queue filled up
    bool stallOnICacheRetry;

    // Holds all outstanding addresses, maps from address to mf object used
    // mostly for acking GPGPU-Sim
    std::map<Addr,OpuMemfetch *> busyInstCacheLineAddrs;

    // Holds instruction packets that need to be retried
    std::list<PacketPtr> retryInstPkts;

    // The TLB to translate instruction addresses
    OpuTLB *itb;

    // Point to GPU this CUDA core is part of
    OpuTop *opuTop;

    // Pointer to GPGPU-Sim shader this CUDA core is a proxy for
    OpuCoreBase *coreImpl;

    // Indicate whether an outstanding memory fence must be signaled back to
    // the shader (e.g. to unblock warp issue)
    std::vector<bool> needsFenceUnblock;
    unsigned maxNumWarpsPerCore;

    // if true then need to signal GPGPU-Sim once cleanup is done
    bool signalKernelFinish;

    // Returns the line of the address, a
    Addr addrToLine(Addr a);

    // Can we issue an inst  cache request this cycle?
    int instCacheResourceAvailable(Addr a);

    Cycles lastActiveCycle;

    std::map<unsigned, bool> coreCTAActive;
    std::map<unsigned, std::vector<Tick> > coreCTAActiveStats;
    Cycles beginActiveCycle;
    int activeCTAs;

  public:
    // Constructor and deconstructor
    OpuCore(const OpuCoreParams &p);
    ~OpuCore();

    // Required for implementing ClockedObject
    // TODO schi change from BaseMasterPort
    virtual Port& getPort(const std::string &if_name,
                                          PortID idx = -1);

    // For checkpoint restore (empty unserialize)
    virtual void unserialize(CheckpointIn &cp);

    // Perform initialization. Called from SPA
    void initialize();

    /** This function is used by the page table walker to determine if it could
     * translate the a pending request or if the underlying request has been
     * squashed. This always returns false for the GPU as it never
     * executes any instructions speculatively.
     * @ return Is the current instruction squashed?
     */
    bool isSquashed() const { return false; }

    // Instruction memory access functions:
    // Initializes an instruction fetch from gem5-side memory
    void icacheFetch(Addr a, OpuMemfetch *mf);

    // Required for translation. Calls sendInstAccess with physical address
    void finishTranslation(WholeTranslationState *state);

  private:
    // Sends an instruction memory access to the cache
    void sendInstAccess(PacketPtr pkt);

    // Handle an instruction port retry request
    void handleRetry();

  public:
    // Receive and complete an instruction fetch
    void recvInstResp(PacketPtr pkt);

    /**
     * This function is the main entrypoint from GPGPU-Sim
     * This function parses the instruction from GPGPU-Sim and issues the
     * memory request to the LSQ on a per-lane basis.
     * @return true if stall
     */
    bool executeMemOp(const OpuWarpinst &inst);

    /**
     * The specified lane is returning a packet from the ShaderLSQ to be
     * handled as appropriate (e.g. LD instructions return data, fences may
     * need to be signaled to the appropriate thread)
     */
    bool recvLSQDataResp(PacketPtr pkt, int lane_id);

    /**
     * The ShaderLSQ is returning a control signal. This currently handles
     * flushes, but may handle other situations that need to block/unblock
     * warp issue.
     */
    void recvLSQControlResp(PacketPtr pkt);

    /**
     * GPGPU-Sim calls this function when the writeback register in its ld/st
     * unit is cleared. If there is a lsqPort blocked, it may now try again
     */
    void writebackClear();

    /**
     * Flush the core of all pending instructions,
     * This is currently used to force the LSQ to flush on kernel end
     */
    void flush();

    /**
     * Called from GPGPU-Sim when a kernel completes on this shader
     * Must signal back to GPGPU-Sim after all cleanup is done
     */
    void finishKernel();

    // For counting statistics
    Stats::Scalar numLocalLoads;
    Stats::Scalar numLocalStores;
    Stats::Scalar numSharedLoads;
    Stats::Scalar numSharedStores;
    Stats::Scalar numParamKernelLoads;
    Stats::Scalar numParamLocalLoads;
    Stats::Scalar numParamLocalStores;
    Stats::Scalar numConstLoads;
    Stats::Scalar numTexLoads;
    Stats::Scalar numGlobalLoads;
    Stats::Scalar numGlobalStores;
    Stats::Scalar numSurfLoads;
    Stats::Scalar numGenericLoads;
    Stats::Scalar numGenericStores;
    Stats::Scalar numDataCacheRequests;
    Stats::Scalar numDataCacheRetry;
    Stats::Scalar numInstCacheRequests;
    Stats::Scalar numInstCacheRetry;
    Stats::Vector instCounts;
    Stats::Scalar activeCycles;
    Stats::Scalar notStalledCycles;
    Stats::Scalar instInstances;
    Stats::Formula instPerCycle;
    Stats::Scalar numKernelsCompleted;
    void regStats();
    void record_ld(opu_mspace_t space);
    void record_st(opu_mspace_t space);
    void record_inst(int inst_type);
    void record_block_issue(unsigned hw_cta_id);
    void record_block_commit(unsigned hw_cta_id);
    void printCTAStats(std::ostream& out);
};

}

#endif

