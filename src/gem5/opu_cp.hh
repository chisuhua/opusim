/*
 * Copyright (c) 2015 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 2002-2005 The Regents of The University of Michigan
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
 *
 */

#ifndef __OPU_COMMAND_PROCESSOR_HH__
#define __OPU_COMMAND_PROCESSOR_HH__

#include <atomic>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>

#include "base/statistics.hh"
#include "cpu/translation.hh"
#include "dev/io_device.hh"
// #include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "opu_tlb.hh"
#include "mem/port.hh"
#include "params/OpuCp.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"
#include "sim/stats.hh"

namespace gem5 {

class OpuTop;

/**
 * The OpuCp class tests a cache coherent memory system by
 * generating false sharing and verifying the read data against a
 * reference updated on the completion of writes. Each tester reads
 * and writes a specific byte in a cache line, as determined by its
 * unique id. Thus, all requests issued by the OpuCp instance are a
 * single byte and a specific address is only ever touched by a single
 * tester.
 *
 * In addition to verifying the data, the tester also has timeouts for
 * both requests and responses, thus checking that the memory-system
 * is making progress.
 */
 enum CpRegIndex
 {
        CP_ID,
        CP_VERSION,
        CP_RPTR,
        CP_WPTR,
        CP_NUM_REGS
 };


class OpuCp : public ClockedObject
{
    uint32_t regs[CP_NUM_REGS];

  public:
    class CPPort : public MasterPort
    {
        friend class OpuCp;

      private:
        OpuCp *cp;

        /// holds packets that failed to send for retry
        std::queue<PacketPtr> outstandingPkts;
        int idx;

      public:
        CPPort(const std::string &_name, OpuCp *_cp, int _idx)
            : MasterPort(_name, _cp), cp(_cp), idx(_idx)
        { }

      protected:
        bool recvTimingResp(PacketPtr pkt);
        // void recvTimingSnoopReq(PacketPtr pkt) { }
        // void recvFunctionalSnoop(PacketPtr pkt) { }
        // Tick recvAtomicSnoop(PacketPtr pkt) { return 0; }
        void recvReqRetry();
        void setStalled(PacketPtr pkt)
        {
            outstandingPkts.push(pkt);
        }
        bool isStalled() { return !outstandingPkts.empty(); }
        void sendPacket(PacketPtr pkt);
    };

    typedef OpuCpParams Params;
    OpuCp(const Params &p);

    CPPort hostPort;
    CPPort devicePort;
    PioPort<OpuCp> pioPort;

    CPPort* writePort;
    CPPort* readPort;
    int driverDelay;
    bool active;     // cuda_gpu set active to true after cpMemory is allocated

    Addr pioAddr;
    Addr pioSize;
    Tick pioDelay;

private:
    OpuTop *opuTop;

    unsigned cacheLineSize;
    unsigned bufferDepth;
    bool buffersFull();
    // Pointers to the actual TLBs
    OpuTLB *hostDTB;
    OpuTLB *deviceDTB;

    // Pointers set as appropriate for memory space during a memcpy
    OpuTLB *readDTB;
    OpuTLB *writeDTB;

    std::atomic<bool> needToRead;
    std::atomic<bool> needToWrite;
    Addr currentReadAddr;
    Addr currentWriteAddr;
    Addr beginAddr;
    Tick writeLeft;
    Tick writeDone;
    Tick readLeft;
    Tick readDone;
    Tick totalLength;

    uint8_t *curData;
    bool *readsDone;

    Tick memAccessStartTime;
    size_t memAccessLength;
    class MemAccessStats {
    public:
        MemAccessStats(Tick _ticks, size_t _bytes) :
            ticks(_ticks), bytes(_bytes)
        { }
        Tick ticks;
        size_t bytes;
    };
    std::vector<MemAccessStats> memAccessStats;

  public:
    std::atomic<bool> running;
    Port &getPort(const std::string &if_name, PortID idx=InvalidPortID) override;
    void finishTranslation(WholeTranslationState *state);

    void finishMemAccess();
    // void memAccess( bool cmd_read, uint32_t data, bool uncacheable, Addr paddr, bool functional);
    void mem_write(Addr addr, uint8_t *p, size_t length, bool host = false);
    void mem_read(Addr addr, uint8_t *p, size_t length, bool host = false) ;

    // store the expected value for the addresses we have touched
    // std::unordered_map<Addr, uint32_t> referenceData;


  protected:
    // method call into zephyr os
    void zephyrOs();

    void zephyrOsTick();
    EventFunctionWrapper zephyrOsTickEvent;

    void memAccessTick();
    EventFunctionWrapper memAccessTickEvent;

    void tryRead();
    void tryWrite();


    PacketPtr retryPkt;


    std::thread *zephyr_thread;
    std::string fileName;


    /** Request id for all generated traffic */
    RequestorID masterId;

    unsigned int id;

    std::set<Addr> outstandingAddrs;

    Tick startTick;
    Tick OSEventTick;

    const bool atomic;

    const bool suppressFuncWarnings;

    Stats::Scalar numReadsStat;
    Stats::Scalar numWritesStat;

    /**
     * Complete a request by checking the response.
     *
     * @param pkt Response packet
     * @param functional Whether the access was functional or not
     */
    void completeRequest(PacketPtr pkt, bool functional = false);

    bool sendPkt(PacketPtr pkt);

  public:
    // Pio
    AddrRangeList getAddrRanges() const;
    uint32_t readReg(CpRegIndex reg);
    void setReg(CpRegIndex reg, uint32_t val);
    CpRegIndex decodeAddr(Addr paddr);

    /*
     * Initialize this object by registering it with the IO APIC.
     */
    void init() override;

    /*
     * Functions to interact with the interrupt port.
     */
    Tick read(PacketPtr pkt);
    Tick write(PacketPtr pkt);

    /** This function is used by the page table walker to determine if it could
    * translate the a pending request or if the underlying request has been
    * squashed. This always returns false for the GPU as it never
    * executes any instructions speculatively.
    * @ return Is the current instruction squashed?
    */
    bool isSquashed() const { return false; }


    void recvRetry();

    Stats::Scalar numOperations;
    Stats::Scalar bytesRead;
    Stats::Scalar bytesWritten;
    Stats::Scalar operationTimeTicks;
    void regStats();

};
}

#endif // __CPU_ZEPHYR_ZEPHYR_HH__
