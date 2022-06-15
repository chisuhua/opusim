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

#ifndef __OPU_COPY_ENGINE_HH__
#define __OPU_COPY_ENGINE_HH__

#include "base/callback.hh"
#include "cpu/translation.hh"
#include "sim/clocked_object.hh"
#include "params/OpuDma.hh"
#include "opu_stream.hh"
// @TODO: Fix the dependencies between sp_array and copy_engine, and
// sort this include into the set above
#include "opu_top.hh"

namespace gem5 {

class OpuDma : public ClockedObject
{
private:
    typedef OpuDmaParams Params;

    void callback() ;

    class CEPort : public RequestPort
    {
        friend class OpuDma;

    private:
        OpuDma *engine;

        /// holds packets that failed to send for retry
        std::queue<PacketPtr> outstandingPkts;

        int idx;

    public:
        CEPort(const std::string &_name, OpuDma *_proc, int _idx)
        : RequestPort(_name, _proc), engine(_proc), idx(_idx) {}

    protected:
        virtual bool recvTimingResp(PacketPtr pkt);
        virtual void recvReqRetry();
        virtual Tick recvAtomic(PacketPtr pkt);
        virtual void recvFunctional(PacketPtr pkt);
        void setStalled(PacketPtr pkt)
        {
            outstandingPkts.push(pkt);
        }
        bool isStalled() { return !outstandingPkts.empty(); }
        void sendPacket(PacketPtr pkt);
    };

    CEPort hostPort;
    CEPort devicePort;

    // Depending on memcpy type, these point to the appropriate ports
    CEPort* readPort;
    CEPort* writePort;

    class TickEvent : public Event
    {
        friend class OpuDma;

    private:
        OpuDma *engine;

    public:
        TickEvent(OpuDma *_engine) : Event(CPU_Tick_Pri), engine(_engine) {}
        void process() { engine->tick(); }
        virtual const char *description() const { return "OpuDma tick"; }
    };

    TickEvent tickEvent;
    RequestorID masterId;

private:
    OpuTop *opuTop;

    unsigned cacheLineSize;
    unsigned bufferDepth;
    bool buffersFull();
    void tick();

    int driverDelay;

    // Pointers to the actual TLBs
    OpuTLB *hostDTB;
    OpuTLB *deviceDTB;

    // Pointers set as appropriate for memory space during a memcpy
    OpuTLB *readDTB;
    OpuTLB *writeDTB;

    bool needToRead;
    bool needToWrite;
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
    bool running;

    void tryRead();
    void tryWrite();
    void finishMemcpy();

    Tick memCpyStartTime;
    size_t memCpyLength;
    class MemCpyStats {
    public:
        MemCpyStats(Tick _ticks, size_t _bytes) :
            ticks(_ticks), bytes(_bytes)
        { }
        Tick ticks;
        size_t bytes;
    };
    std::vector<MemCpyStats> memCpyStats;

public:

    OpuDma(const Params &p);
    // TODO schi change from BaseMasterPort
    virtual Port& getPort(const std::string &if_name, PortID idx = -1);
    void finishTranslation(WholeTranslationState *state);
    int memcpy(Addr src, Addr dst, size_t length, stream_operation_type type);
    int memset(Addr dst, int value, size_t length);
    void recvPacket(PacketPtr pkt);

    /** This function is used by the page table walker to determine if it could
    * translate the a pending request or if the underlying request has been
    * squashed. This always returns false for the GPU as it never
    * executes any instructions speculatively.
    * @ return Is the current instruction squashed?
    */
    bool isSquashed() const { return false; }

    void cePrintStats(std::ostream& out);

    OutputStream* statsFile;

    Stats::Scalar numOperations;
    Stats::Scalar bytesRead;
    Stats::Scalar bytesWritten;
    Stats::Scalar operationTimeTicks;
    void regStats();
};
}

#endif
