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

#include <cmath>
#include <iostream>
#include <map>

#include "cpu/translation.hh"
#include "debug/OpuCore.hh"
#include "debug/OpuCoreAccess.hh"
#include "debug/OpuCoreFetch.hh"
#include "opu_core.hh"
#include "opu_top.hh"
#include "mem/page_table.hh"
#include "params/OpuCore.hh"
#include "sim/system.hh"

using namespace std;

namespace gem5 {

// FIXME schi add this to getMasterId, anem remove name()
OpuCore::OpuCore(const OpuCoreParams &p) :
    ClockedObject(p), instPort(name() + ".inst_port", this),
    lsqControlPort(name() + ".lsq_ctrl_port", this), _params(&p),
    // dataMasterId(p->sys->getMasterId(this, name() + ".data")),
    // instMasterId(p->sys->getMasterId(this, name() + ".inst")), id(p->id),
    dataMasterId(p.sys->getRequestorId(this, "data")),
    instMasterId(p.sys->getRequestorId(this, "inst")), id(p.id),
    itb(p.itb), opuTop(p.gpu), maxNumWarpsPerCore(p.warp_contexts)
{
    writebackBlocked = -1; // Writeback is not blocked

    stallOnICacheRetry = false;

    opuTop->registerOpuCore(this);

    warpSize = opuTop->getWarpSize();

    signalKernelFinish = false;

    if (p.port_lsq_port_connection_count != warpSize) {
        panic("Shader core lsq_port size != to warp size\n");
    }

    // create the ports
    for (int i = 0; i < warpSize; ++i) {
        lsqPorts.push_back(new LSQPort(csprintf("%s-lsqPort%d", name(), i),
                                    this, i));
    }

    activeCTAs = 0;

    needsFenceUnblock.resize(maxNumWarpsPerCore);
    for (int i = 0; i < maxNumWarpsPerCore; i++) {
        needsFenceUnblock[i] = false;
    }
}

OpuCore::~OpuCore()
{
    for (int i = 0; i < warpSize; ++i) {
        delete lsqPorts[i];
    }
    lsqPorts.clear();
}

// TODO schi change
// BaseMasterPort&
Port&
OpuCore::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "inst_port") {
        return instPort;
    } else if (if_name == "lsq_port") {
        if (idx >= static_cast<PortID>(lsqPorts.size())) {
            panic("OpuCore::getMasterPort: unknown index %d\n", idx);
        }
        return *lsqPorts[idx];
    } else if (if_name == "lsq_ctrl_port") {
        return lsqControlPort;
    } else {
        // TODO schi return MemObject::getMasterPort(if_name, idx);
        return ClockedObject::getPort(if_name, idx);
    }
}

void
OpuCore::unserialize(CheckpointIn &cp)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}

void OpuCore::initialize()
{
    coreImpl = opuTop->getOpuUsim()->getSIMTCore(id);
    coreImpl->setup_cb_icacheFetch(std::bind(&OpuCore::icacheFetch, this, std::placeholders::_1, std::placeholders::_2));
    coreImpl->setup_cb_getLocalBaseVaddr(std::bind(&OpuTop::getLocalBaseVaddr, opuTop));
}

int OpuCore::instCacheResourceAvailable(Addr addr)
{
    map<Addr,OpuMemfetch *>::iterator iter =
            busyInstCacheLineAddrs.find(addrToLine(addr));
    return iter == busyInstCacheLineAddrs.end();
}

inline Addr OpuCore::addrToLine(Addr a)
{
    unsigned int maskBits = opuTop->getRubySystem()->getBlockSizeBits();
    return a & (((uint64_t)-1) << maskBits);
}

void
OpuCore::icacheFetch(Addr addr, OpuMemfetch *mf)
{
    assert(instCacheResourceAvailable(addr));

    Addr line_addr = addrToLine(addr);
    DPRINTF(OpuCoreFetch,
            "Fetch request, addr: 0x%x, size: %d, line: 0x%x\n",
            addr, mf->size(), line_addr);
/* TODO schi change
    RequestPtr req = new Request();
    */
    RequestPtr req = std::make_shared<Request>();
    Request::Flags flags;
    Addr pc = (Addr)mf->get_pc();
    const int asid = 0;

    BaseMMU::Mode mode = BaseMMU::Read;
    req->setVirt(asid, line_addr, mf->size(), flags, instMasterId, pc);
    req->setFlags(Request::INST_FETCH);

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<OpuCore*> *translation
            = new DataTranslation<OpuCore*>(this, state);

    busyInstCacheLineAddrs[addrToLine(req->getVaddr())] = mf;
    itb->beginTranslateTiming(req, translation, mode);
}

void OpuCore::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Instruction translation encountered fault (%s) for address 0x%x",
              state->getFault()->name(), state->mainReq->getVaddr());
    }
    assert(state->mode == BaseMMU::Read);
    PacketPtr pkt = new Packet(state->mainReq, MemCmd::ReadReq);
    pkt->allocate();
    assert(pkt->req->isInstFetch());
    if (!stallOnICacheRetry) {
        sendInstAccess(pkt);
    } else {
        DPRINTF(OpuCoreFetch, "Port blocked, add vaddr: 0x%x to retry list\n",
                pkt->req->getVaddr());
        retryInstPkts.push_back(pkt);
    }
    delete state;
}

void
OpuCore::sendInstAccess(PacketPtr pkt)
{
    assert(!stallOnICacheRetry);

    DPRINTF(OpuCoreFetch,
            "Sending inst read of %d bytes to vaddr: 0x%x\n",
            pkt->getSize(), pkt->req->getVaddr());

    if (!instPort.sendTimingReq(pkt)) {
        stallOnICacheRetry = true;
        if (pkt != retryInstPkts.front()) {
            retryInstPkts.push_back(pkt);
        }
        DPRINTF(OpuCoreFetch, "Send failed vaddr: 0x%x. Waiting: %d\n",
                pkt->req->getVaddr(), retryInstPkts.size());
    }
    numInstCacheRequests++;
}

void
OpuCore::handleRetry()
{
    assert(stallOnICacheRetry);
    assert(retryInstPkts.size());

    numInstCacheRetry++;

    PacketPtr retry_pkt = retryInstPkts.front();
    DPRINTF(OpuCoreFetch, "Received retry, vaddr: 0x%x\n",
            retry_pkt->req->getVaddr());

    if (instPort.sendTimingReq(retry_pkt)) {
        retryInstPkts.remove(retry_pkt);

        // If there are still packets on the retry list, signal to Ruby that
        // there should be a retry call for the next packet when possible
        stallOnICacheRetry = (retryInstPkts.size() > 0);
        if (stallOnICacheRetry) {
            retry_pkt = retryInstPkts.front();
            instPort.sendTimingReq(retry_pkt);
        }
    } else {
        panic("Access should never fail on a retry!");
    }
}

void
OpuCore::recvInstResp(PacketPtr pkt)
{
    assert(pkt->req->isInstFetch());
    map<Addr,OpuMemfetch *>::iterator iter =
            busyInstCacheLineAddrs.find(addrToLine(pkt->req->getVaddr()));
    assert(iter != busyInstCacheLineAddrs.end());

    DPRINTF(OpuCoreFetch, "Finished fetch on vaddr 0x%x\n",
            pkt->req->getVaddr());

    coreImpl->accept_fetch_response(iter->second);

    busyInstCacheLineAddrs.erase(iter);

    // TODO schi if (pkt->req) delete pkt->req;
    delete pkt;
}

bool
OpuCore::executeMemOp(const OpuWarpinst &inst)
{
    assert(inst.space_type == opu_mspace_t::GLOBAL ||
           // inst.space_type == const_space ||
           inst.space_type == opu_mspace_t::PRIVATE ||
           inst.op == opu_op_t::BARRIER_OP ||
           inst.op == opu_op_t::MEMORY_BARRIER_OP);
    assert(inst.valid());

    // for debugging
    bool completed = false;

    int size = inst.data_size;
    if (inst.is_load() || inst.is_store()) {
        assert(size >= 1 && size <= 8);
    }
    size *= inst.vectorLength;
    assert(size <= 16);
    if (inst.op == opu_op_t::BARRIER_OP || inst.op == opu_op_t::MEMORY_BARRIER_OP) {
        if (inst.active_count() != inst.warp_size()) {
            warn_once("ShaderLSQ received partial-warp fence: Assuming you know what you're doing");
        }
    }
    const int asid = 0;
    Request::Flags flags;
    if (inst.isatomic()) {
        assert(inst.memory_op == opu_memop_t::STORE);
        // Assert that gem5-gpu knows how to handle the requested atomic type.
        // TODO: When all atomic types and data sizes are implemented, remove
        assert(inst.get_atomic() == opu_atomic_t::ATOMIC_INC ||
               inst.get_atomic() == opu_atomic_t::ATOMIC_MAX ||
               inst.get_atomic() == opu_atomic_t::ATOMIC_MIN ||
               inst.get_atomic() == opu_atomic_t::ATOMIC_ADD ||
               inst.get_atomic() == opu_atomic_t::ATOMIC_CAS);
        assert(inst.data_type == opu_datatype_t::DT_INT32 ||
               inst.data_type == opu_datatype_t::DT_UINT32 ||
               inst.data_type == opu_datatype_t::DT_FP32 ||
               inst.data_type == opu_datatype_t::DT_B32);
        // GPU atomics will use the MEM_SWAP flag to indicate to Ruby that the
        // request should be passed to the cache hierarchy as secondary
        // RubyRequest_Atomic.
        // NOTE: Most GPU atomics have conditional writes and most perform some
        //       operation on loaded data before writing it back into cache.
        //       This makes them read-modify-conditional-write operations, but
        //       for ease of development, use the MEM_SWAP designator for now.
        flags.set(Request::MEM_SWAP);
    }

#if 0
    if (inst.space_type == const_space) {
        DPRINTF(OpuCoreAccess, "Const space: %p\n", inst.pc);
    } else if (inst.space_type == local_space) {
        DPRINTF(OpuCoreAccess, "Local space: %p\n", inst.pc);
    } else if (inst.space_type == param_space_local) {
        DPRINTF(OpuCoreAccess, "Param local space: %p\n", inst.pc);
    } else {
        DPRINTF(OpuCoreAccess, "Global space: %p\n", inst.pc);
    }
#endif

    for (int lane = 0; lane < warpSize; lane++) {
        if (inst.active(lane)) {
            Addr addr = inst.get_addr(lane);

            PacketPtr pkt;
            if (inst.is_load()) {
                // Not all cache operators are currently supported in gem5-gpu.
                // Verify that a supported cache operator is specified for this
                // load instruction.
                if (!inst.isatomic() && inst.cache_op == opu_cacheop_t::CACHE_GLOBAL) {
                    // If this is a load instruction that must access coherent
                    // global memory, bypass the L1 cache to avoid stale hits
                    flags.set(Request::BYPASS_L1);
                } else if (inst.cache_op != opu_cacheop_t::CACHE_ALL &&
                    !(inst.isatomic() && inst.cache_op == opu_cacheop_t::CACHE_GLOBAL)) {
                    panic("Unhandled cache operator (%d) on load\n",
                          static_cast<int>(inst.cache_op));
                }
                // TODO schi RequestPtr req = new Request(asid, addr, size, flags,
                RequestPtr req = std::make_shared<Request>(asid, addr, size, flags,
                        dataMasterId, inst.pc, id, inst.warp_id);
                pkt = new Packet(req, MemCmd::ReadReq);
                if (inst.isatomic()) {
                    assert(flags.isSet(Request::MEM_SWAP));
                    AtomicOpRequest *pkt_data = new AtomicOpRequest();
                    pkt_data->lastAccess = true;
                    pkt_data->uniqueId = lane;
                    pkt_data->dataType = getDataType(inst.data_type);
                    pkt_data->atomicOp = getAtomOpType(inst.get_atomic());
                    pkt_data->lineOffset = 0;
                    pkt_data->setData((uint8_t*)inst.get_data(lane));

                    // TODO: If supporting atomics that require more operands,
                    // will need to copy that data here also

                    // Create packet data to include the atomic type and
                    // the register data to be used (e.g. atomicInc requires
                    // the saturating value up to which to count)
                    pkt->dataDynamic(pkt_data);
                } else {
                    pkt->allocate();
                }
                // Since only loads return to the OpuCore
                pkt->senderState = new SenderState(inst);
            } else if (inst.is_store()) {
                assert(!inst.isatomic());
                // Not all cache operators are currently supported in gem5-gpu.
                // Verify that a supported cache operator is specified for this
                // load instruction.
                if (inst.cache_op == opu_cacheop_t::CACHE_GLOBAL) {
                    flags.set(Request::BYPASS_L1);
                } else if (inst.cache_op != opu_cacheop_t::CACHE_ALL &&
                           inst.cache_op != opu_cacheop_t::CACHE_WRITE_BACK) {
                    panic("Unhandled cache operator (%d) on store\n",
                          static_cast<int>(inst.cache_op));
                }
                // TODO schi RequestPtr req = new Request(asid, addr, size, flags,
                RequestPtr req = std::make_shared<Request>(asid, addr, size, flags,
                        dataMasterId, inst.pc, id, inst.warp_id);
                pkt = new Packet(req, MemCmd::WriteReq);
                pkt->allocate();
                pkt->setData((uint8_t*)inst.get_data(lane));
                DPRINTF(OpuCoreAccess, "Send store from lane %d address 0x%llx: data = %d\n",
                        lane, pkt->req->getVaddr(), *(int*)inst.get_data(lane));
            } else if (inst.op == opu_op_t::BARRIER_OP || inst.op == opu_op_t::MEMORY_BARRIER_OP) {
                assert(!inst.isatomic());
                // Setup Fence packet
                // TODO: If adding fencing functionality, specify control data
                // in packet or request
                // TODO schi RequestPtr req = new Request(asid, 0x0, 0, flags, dataMasterId,
                RequestPtr req = std::make_shared<Request>(asid, 0x0, 0, flags, dataMasterId,
                        inst.pc, id, inst.warp_id);
                pkt = new Packet(req, MemCmd::FenceReq);
                pkt->senderState = new SenderState(inst);
            } else {
                panic("Unsupported instruction type\n");
            }

            if (!lsqPorts[lane]->sendTimingReq(pkt)) {
                // NOTE: This should fail early. If executeMemOp fails after
                // some, but not all, of the requests have been sent the
                // behavior is undefined.
                if (completed) {
                    panic("Should never fail after first accepted lane");
                }

                if (inst.is_load() || inst.op == opu_op_t::BARRIER_OP ||
                    inst.op == opu_op_t::MEMORY_BARRIER_OP) {
                    delete pkt->senderState;
                }
                // TODO schi delete pkt->req;
                delete pkt;

                // Return that there is a pipeline stall
                return true;
            } else {
                completed = true;
            }
        }
    }

    if (inst.op == opu_op_t::BARRIER_OP || inst.op == opu_op_t::MEMORY_BARRIER_OP) {
        needsFenceUnblock[inst.warp_id] = true;
    }

    // Return that there should not be a pipeline stall
    return false;
}

bool
OpuCore::recvLSQDataResp(PacketPtr pkt, int lane_id)
{
    assert(pkt->isRead() || pkt->cmd == MemCmd::FenceResp);

    DPRINTF(OpuCoreAccess, "Got a response for lane %d address 0x%llx\n",
            lane_id, pkt->req->getVaddr());

    const OpuWarpinst &inst = ((SenderState*)pkt->senderState)->inst;
    assert(!inst.empty() && inst.valid());

    if (pkt->isRead()) {
        if (!coreImpl->ldst_unit_wb_inst(inst)) {
            // Writeback register is occupied, stall
            assert(writebackBlocked < 0);
            writebackBlocked = lane_id;
            return false;
        }

        uint8_t data[16];
        assert(pkt->getSize() <= sizeof(data));

        if (inst.isatomic()) {
            assert(pkt->req->isSwap());
            AtomicOpRequest *lane_req = pkt->getPtr<AtomicOpRequest>();
            lane_req->writeData(data);
        } else {
            pkt->writeData(data);
        }
        DPRINTF(OpuCoreAccess, "Loaded data %d\n", *(int*)data);
        coreImpl->writeRegister(inst, warpSize, lane_id, (char*)data);
    } else if (pkt->cmd == MemCmd::FenceResp) {
        if (needsFenceUnblock[inst.warp_id]) {
            if (inst.op == opu_op_t::BARRIER_OP) {
                // Signal that warp has reached barrier
                assert(!coreImpl->warp_waiting_at_barrier(inst.warp_id));
                coreImpl->warp_reaches_barrier(inst);
                DPRINTF(OpuCoreAccess, "Warp %d reaches barrier\n",
                        pkt->req->threadId());
            }

            // Signal that fence has been cleared
            assert(coreImpl->fence_unblock_needed(inst.warp_id));
            coreImpl->complete_fence(pkt->req->threadId());
            DPRINTF(OpuCoreAccess, "Cleared fence, unblocking warp %d\n",
                    pkt->req->threadId());

            needsFenceUnblock[inst.warp_id] = false;
        }
    }

    // TODO schi delete pkt->senderState;
    // delete pkt->req;
    delete pkt;

    return true;
}

void
OpuCore::recvLSQControlResp(PacketPtr pkt)
{
    if (pkt->isFlush()) {
        DPRINTF(OpuCoreAccess, "Got flush response\n");
        if (signalKernelFinish) {
            coreImpl->finish_kernel();
            signalKernelFinish = false;
        }
    } else {
        panic("Received unhandled packet type in control port");
    }
    // TODO schi delete pkt->req;
    delete pkt;
}

void
OpuCore::writebackClear()
{
    if (writebackBlocked >= 0) lsqPorts[writebackBlocked]->sendRetryResp();
    writebackBlocked = -1;
}

void
OpuCore::flush()
{
    int asid = 0;
    Addr addr(0);
    Request::Flags flags;
    // TODO schi RequestPtr req = new Request(asid, addr, flags, dataMasterId);
    RequestPtr req = std::make_shared<Request>(asid, addr, flags, dataMasterId);
    PacketPtr pkt = new Packet(req, MemCmd::FlushReq);

    DPRINTF(OpuCoreAccess, "Sending flush request\n");
    if (!lsqControlPort.sendTimingReq(pkt)){
        panic("Flush requests should never fail");
    }
}

void
OpuCore::finishKernel()
{
    numKernelsCompleted++;
    signalKernelFinish = true;
    flush();
}

bool
OpuCore::LSQPort::recvTimingResp(PacketPtr pkt)
{
    return core->recvLSQDataResp(pkt, idx);
}

void
OpuCore::LSQPort::recvReqRetry()
{
    panic("Not sure how to respond to a recvReqRetry...");
}

bool
OpuCore::LSQControlPort::recvTimingResp(PacketPtr pkt)
{
    core->recvLSQControlResp(pkt);
    return true;
}

void
OpuCore::LSQControlPort::recvReqRetry()
{
    panic("OpuCore::LSQControlPort::recvReqRetry() not implemented!");
}

bool
OpuCore::InstPort::recvTimingResp(PacketPtr pkt)
{
    core->recvInstResp(pkt);
    return true;
}

void
OpuCore::InstPort::recvReqRetry()
{
    core->handleRetry();
}

Tick
OpuCore::InstPort::recvAtomic(PacketPtr pkt)
{
    panic("Not sure how to recvAtomic");
    return 0;
}

void
OpuCore::InstPort::recvFunctional(PacketPtr pkt)
{
    panic("Not sure how to recvFunctional");
}

/*
OpuCore *OpuCoreParams::create() const {
    return new OpuCore(*this);
}
*/

void
OpuCore::regStats()
{
    ClockedObject::regStats();
    using namespace Stats;

    numLocalLoads
        .name(name() + ".local_loads")
        .desc("Number of loads from local space")
        ;
    numLocalStores
        .name(name() + ".local_stores")
        .desc("Number of stores to local space")
        ;
    numSharedLoads
        .name(name() + ".shared_loads")
        .desc("Number of loads from shared space")
        ;
    numSharedStores
        .name(name() + ".shared_stores")
        .desc("Number of stores to shared space")
        ;
    numParamKernelLoads
        .name(name() + ".param_kernel_loads")
        .desc("Number of loads from kernel parameter space")
        ;
    numParamLocalLoads
        .name(name() + ".param_local_loads")
        .desc("Number of loads from local parameter space")
        ;
    numParamLocalStores
        .name(name() + ".param_local_stores")
        .desc("Number of stores to local parameter space")
        ;
    numConstLoads
        .name(name() + ".const_loads")
        .desc("Number of loads from constant space")
        ;
    numTexLoads
        .name(name() + ".tex_loads")
        .desc("Number of loads from texture space")
        ;
    numGlobalLoads
        .name(name() + ".global_loads")
        .desc("Number of loads from global space")
        ;
    numGlobalStores
        .name(name() + ".global_stores")
        .desc("Number of stores to global space")
        ;
    numSurfLoads
        .name(name() + ".surf_loads")
        .desc("Number of loads from surface space")
        ;
    numGenericLoads
        .name(name() + ".generic_loads")
        .desc("Number of loads from generic spaces (global, shared, local)")
        ;
    numGenericStores
        .name(name() + ".generic_stores")
        .desc("Number of stores to generic spaces (global, shared, local)")
        ;
    numInstCacheRequests
        .name(name() + ".inst_cache_requests")
        .desc("Number of instruction cache requests sent")
        ;
    numInstCacheRetry
        .name(name() + ".inst_cache_retries")
        .desc("Number of instruction cache retries")
        ;
    instCounts
        .init(8)
        .name(name() + ".inst_counts")
        .desc("Inst counts: 1: ALU, 2: MAD, 3: CTRL, 4: SFU, 5: MEM, 6: TEX, 7: NOP")
        ;

    activeCycles
        .name(name() + ".activeCycles")
        .desc("Number of cycles this shader was executing a CTA")
        ;
    notStalledCycles
        .name(name() + ".notStalledCycles")
        .desc("Number of cycles this shader was actually executing at least one instance")
        ;
    instInstances
        .name(name() + ".instInstances")
        .desc("Total instructions executed by all PEs in the core")
        ;
    instPerCycle
        .name(name() + ".instPerCycle")
        .desc("Instruction instances per cycle")
        ;

    instPerCycle = instInstances / activeCycles;
    numKernelsCompleted
        .name(name() + ".kernels_completed")
        .desc("Number of kernels completed")
        ;

    itb->regStats();
}

void
OpuCore::record_ld(opu_mspace_t space_type)
{
    switch(space_type) {
        case opu_mspace_t::PRIVATE: numLocalLoads++; break;
        case opu_mspace_t::SHARED: numSharedLoads++; break;
        case opu_mspace_t::GLOBAL: numGlobalLoads++; break;
#if 0
    case param_space_kernel: numParamKernelLoads++; break;
    case param_space_local: numParamLocalLoads++; break;
    case const_space: numConstLoads++; break;
    case tex_space: numTexLoads++; break;
    case surf_space: numSurfLoads++; break;
    case generic_space: numGenericLoads++; break;
    case param_space_unclassified:
    case undefined_space:
    case reg_space:
    case instruction_space:
#endif
    default:
        panic("Load from invalid space: %d!", static_cast<int>(space_type));
        break;
    }
}

void
OpuCore::record_st(opu_mspace_t space_type)
{
    switch(space_type) {
        case opu_mspace_t::PRIVATE: numLocalStores++; break;
        case opu_mspace_t::SHARED: numSharedStores++; break;
        case opu_mspace_t::GLOBAL: numGlobalStores++; break;
#if 0
    case param_space_local: numParamLocalStores++; break;
    case generic_space: numGenericStores++; break;

    case param_space_kernel:
    case const_space:
    case tex_space:
    case surf_space:
    case param_space_unclassified:
    case undefined_space:
    case reg_space:
    case instruction_space:
#endif
    default:
        panic("Store to invalid space: %d!", static_cast<int>(space_type));
        break;
    }
}

void
OpuCore::record_inst(int inst_type)
{
    instCounts[inst_type]++;

    // if not nop
    if (inst_type != 7) {
        instInstances++;
        if (curCycle() != lastActiveCycle) {
            lastActiveCycle = curCycle();
            notStalledCycles++;
        }
    }
}

void
OpuCore::record_block_issue(unsigned hw_cta_id)
{
    assert(!coreCTAActive[hw_cta_id]);
    coreCTAActive[hw_cta_id] = true;
    coreCTAActiveStats[hw_cta_id].push_back(curTick());

    if (activeCTAs == 0) {
        beginActiveCycle = curCycle();
    }
    activeCTAs++;
}

void
OpuCore::record_block_commit(unsigned hw_cta_id)
{
    assert(coreCTAActive[hw_cta_id]);
    coreCTAActive[hw_cta_id] = false;
    coreCTAActiveStats[hw_cta_id].push_back(curTick());

    activeCTAs--;
    if (activeCTAs == 0) {
        activeCycles += curCycle() - beginActiveCycle;
    }
}

void OpuCore::printCTAStats(std::ostream& out)
{
    std::map<unsigned, std::vector<Tick> >::iterator iter =
            coreCTAActiveStats.begin();
    std::vector<Tick>::iterator times;
    for (; iter != coreCTAActiveStats.end(); iter++) {
        unsigned cta_id = iter->first;
        out << id << ", " << cta_id << ", ";
        times = coreCTAActiveStats[cta_id].begin();
        for (; times != coreCTAActiveStats[cta_id].end(); times++) {
            out << *times << ", ";
        }
        out << curTick() << "\n";
    }
}

}
