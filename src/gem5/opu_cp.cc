/*
 * Copyright (c) 2015, 2019 ARM Limited
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

#include <atomic>

#include "base/random.hh"
#include "base/statistics.hh"
#include "base/trace.hh"
#include "debug/OpuCp.hh"
#include "opu_cp.hh"
#include "gpu/dso_library.hh"
#include "opu_top.hh"
#include "sim/sim_exit.hh"
#include "sim/stats.hh"
#include "sim/system.hh"

using namespace std;

namespace gem5 {
/*
bool
OpuCp::sendPkt(PacketPtr pkt) {
    if (atomic) {
        port.sendAtomic(pkt);
        completeRequest(pkt);
    } else {
        if (!port.sendTimingReq(pkt)) {
            retryPkt = pkt;
            return false;
        }
    }
    return true;
}
*/

template <typename T>
std::function<T> GetFunction(void * func)
{
    return std::function<T>((T*)(func));
}

static OpuCp *zephyr_instance;
// static EventQueue *event_queue;
// static atomic<Tick> os_tick;
static Tick os_tick;
// __thread std::condition_variable data_read_done;
//
static std::mutex os_tick_mutex;
// static atomic<bool> write_done_flag;
// static atomic<bool> read_done_flag;


static void zephyr_api_write_blob(Addr dst, uint8_t *p, size_t length, bool host = false) {
    zephyr_instance->mem_write(dst, p, length, host);
    while (zephyr_instance->running) {
        std::this_thread::yield();
    }
}

static void zephyr_api_read_blob(Addr src, uint8_t *p, size_t length, bool host = false) {
    zephyr_instance->mem_read(src, p, length, host);
    while (zephyr_instance->running) {
        std::this_thread::yield();
    }
}

// TODO add  api to exit sim
static void zephyr_api_exit_sim() {
    exitSimLoop("maximum number of loads reached");
}

void OpuCp::mem_write(Addr addr, uint8_t *p, size_t length, bool host) {
    std::lock_guard<std::mutex> lock(os_tick_mutex);
    assert(length > 0);
    assert(!running);
    running = true;
    needToWrite = true;
    needToRead = false;

    memAccessLength = length;
    currentWriteAddr = addr;
    writeLeft = length;
    totalLength = length;
    writeDone = 0;
    curData = p;
    if (host) {
        writePort = &hostPort;
        writeDTB = hostDTB;
    } else {
        writePort = &devicePort;
        writeDTB = deviceDTB;
    }
}

void OpuCp::mem_read(Addr addr, uint8_t *p, size_t length, bool host) {
    std::lock_guard<std::mutex> lock(os_tick_mutex);
    assert(length > 0);
    assert(!running);
    running = true;
    needToWrite = false;
    needToRead = true;

    memAccessLength = length;
    currentReadAddr = addr;
    beginAddr = addr;
    readLeft = length;
    totalLength = length;
    readDone = 0;
    curData = p;
    readsDone = new bool[length];
    for (int i = 0; i < length; i++) {
        curData[i] = 0;
        readsDone[i] = false;
    }
    if (host) {
        readPort = &hostPort;
        readDTB = hostDTB;
    } else {
        readPort = &devicePort;
        readDTB = deviceDTB;
    }
}

void OpuCp::memAccessTick()
{
    if (!running) return;
    if (needToRead && readPort->isStalled()) {
        DPRINTF(OpuCp, "read port Stalled\n");
        schedule(memAccessTickEvent, nextCycle());
    } else if (needToWrite && writePort->isStalled()) {
        DPRINTF(OpuCp, "write port Stalled\n");
        schedule(memAccessTickEvent, nextCycle());
    } else if (needToRead || needToWrite) {
        if (needToRead && !readPort->isStalled() && !buffersFull()) {
            DPRINTF(OpuCp, "trying read\n");
            tryRead();
        }
        // if (needToWrite && !writePort->isStalled() && ((totalLength - writeLeft) < readDone)) {
        if (needToWrite && !writePort->isStalled()) {
            DPRINTF(OpuCp, "trying write\n");
            tryWrite();
        }
    } else {
       DPRINTF(OpuCp, "No needToRead and No need ToWrite, why here ?\n");
       // running = false;
    }
}

void OpuCp::tryRead()
{
    RequestPtr req = std::make_shared<Request>();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    //unsigned block_size = port.peerBlockSize();

    if (readLeft <= 0) {
        DPRINTF(OpuCp, "WHY ARE WE HERE?\n");
        return;
    }

    int size;
    if (currentReadAddr % cacheLineSize) {
        size = cacheLineSize - (currentReadAddr % cacheLineSize);
        DPRINTF(OpuCp, "Aligning\n");
    } else {
        size = cacheLineSize;
    }
    size = readLeft > (size - 1) ? size : readLeft;
    req->setVirt(asid, currentReadAddr, size, flags, masterId, pc);

    DPRINTF(OpuCp, "trying read addr: 0x%x, %d bytes\n", currentReadAddr, size);

    BaseMMU::Mode mode = BaseMMU::Read;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<OpuCp*> *translation
            = new DataTranslation<OpuCp*>(this, state);

    readDTB->beginTranslateTiming(req, translation, mode);

    currentReadAddr += size;

    readLeft -= size;

    if (!(readLeft > 0)) {
        if (readPort->isStalled() && !memAccessTickEvent.scheduled()) {
            schedule(memAccessTickEvent, nextCycle());
        } else {
            needToRead = false;
        }
    } else {
        schedule(memAccessTickEvent, nextCycle());
    }
}

void OpuCp::tryWrite()
{
    if (writeLeft <= 0) {
        DPRINTF(OpuCp, "WHY ARE WE HERE (write)?\n");
        return;
    }

    int size;
    if (currentWriteAddr % cacheLineSize) {
        size = cacheLineSize - (currentWriteAddr % cacheLineSize);
        DPRINTF(OpuCp, "Aligning\n");
    } else {
        size = cacheLineSize;
    }
    size = writeLeft > size-1 ? size : writeLeft;

    /*
    if (readDone < size+(totalLength-writeLeft)) {
        // haven't read enough yet
        DPRINTF(OpuCp, "Tried to write when we haven't read enough\n");
        return;
    }
    */

    RequestPtr req = std::make_shared<Request>();
    Request::Flags flags;
    Addr pc = 0;
    const int asid = 0;
    req->setVirt(asid, currentWriteAddr, size, flags, masterId, pc);

    // assert(	(totalLength-writeLeft +size) <= readDone);
    uint8_t *data = new uint8_t[size];
    std::memcpy(data, &curData[totalLength-writeLeft], size);
    req->setExtraData((uint64_t)data);

    DPRINTF(OpuCp, "trying write addr: 0x%x, %d bytes, data %x\n", currentWriteAddr, size, *((int*)(&curData[totalLength-writeLeft])));

    BaseMMU::Mode mode = BaseMMU::Write;

    WholeTranslationState *state =
            new WholeTranslationState(req, NULL, NULL, mode);
    DataTranslation<OpuCp*> *translation
            = new DataTranslation<OpuCp*>(this, state);

    writeDTB->beginTranslateTiming(req, translation, mode);

    currentWriteAddr += size;

    writeLeft -= size;

    if (!(writeLeft > 0)) {
        if (writePort->isStalled() && !memAccessTickEvent.scheduled()) {
            schedule(memAccessTickEvent, nextCycle());
        } else {
            needToWrite = false;
        }
    } else {
        schedule(memAccessTickEvent, nextCycle());
    }

}

bool OpuCp::buffersFull() {
    // unsigned amount_buffered = readDone - (totalLength - writeLeft);
    // FIXME
    unsigned amount_buffered = readDone;
    return (bufferDepth > 0) && (amount_buffered > bufferDepth);
}

void OpuCp::finishTranslation(WholeTranslationState *state)
{
    if (state->getFault() != NoFault) {
        panic("Translation encountered fault (%s) for address 0x%x", state->getFault()->name(), state->mainReq->getVaddr());
    }
    DPRINTF(OpuCp, "Finished translation of Vaddr 0x%x -> Paddr 0x%x\n", state->mainReq->getVaddr(), state->mainReq->getPaddr());
    PacketPtr pkt;
    if (state->mode == BaseMMU::Read) {
        pkt = new Packet(state->mainReq, MemCmd::ReadReq);
        pkt->allocate();
        readPort->sendPacket(pkt);
    } else if (state->mode == BaseMMU::Write) {
        pkt = new Packet(state->mainReq, MemCmd::WriteReq);
        uint8_t *pkt_data = (uint8_t *)state->mainReq->getExtraData();
        pkt->dataDynamic(pkt_data);
        writePort->sendPacket(pkt);
    } else {
        panic("Finished translation of unknown mode: %d\n", state->mode);
    }
    delete state;
}


OpuCp::OpuCp(const Params &p)
    : ClockedObject(p),
      hostPort(name() + ".host_port", this, 0),
      devicePort(name() + ".device_port", this, 0),
      pioPort(this),
      writePort(NULL), readPort(NULL),
      driverDelay(p.driver_delay),
      active(false),
      pioAddr(p.pio_addr),
      pioSize(4096),
      pioDelay(p.pio_latency),
      opuTop(p.opu),
      cacheLineSize(p.cache_line_size),
      hostDTB(p.host_dtb),
      deviceDTB(p.device_dtb),
      readDTB(NULL), writeDTB(NULL),
      zephyrOsTickEvent([this]{ zephyrOsTick(); }, name()),
      memAccessTickEvent([this]{ memAccessTick(); }, name()),
      retryPkt(nullptr),
      masterId(p.system->getRequestorId(this)),
      startTick(p.start_tick),
      OSEventTick(p.os_event_tick),
      atomic(p.system->isAtomicMode()),
      suppressFuncWarnings(p.suppress_func_warnings)
{

    DPRINTF(OpuCp, "Created copy engine\n");

    opuTop->registerCp(this);

    needToRead = false;
    needToWrite = false;
    running = false;

    bufferDepth = p.buffering * cacheLineSize;

    zephyr_instance = this;
    // event_queue = new EventQueue("OpuCpOsEventQueue");

    fileName = p.file_name;

    /*
    schedule(zephyrOsTickEvent, curTick() + startTick);
    zephyr_thread = new std::thread([this](){ this->zephyrOs();});
    zephyr_thread->detach();
    */
}

static void* gem5api_vector[] = {(void*)zephyr_api_read_blob, (void*)zephyr_api_write_blob, (void*)zephyr_api_exit_sim};

void OpuCp::zephyrOsTick()
{
    std::lock_guard<std::mutex> lock(os_tick_mutex);
    os_tick = curTick();

    if (active && running && !memAccessTickEvent.scheduled()) {
        memAccessStartTime = os_tick;
        DPRINTF(OpuCp, "OpuCp OSTick %ld and schedule memAccessTick %ld\n", os_tick, memAccessStartTime);
        schedule(memAccessTickEvent, nextCycle() + driverDelay);
    }

    schedule(zephyrOsTickEvent, os_tick + OSEventTick);

}


void OpuCp::zephyrOs()
{
    DSOLibrary lib;
    lib.Init(fileName);

    typedef int(*pFunc)(int, char *[]);

    int argc = 2;
    char argv_str[] = "gem5_main";
    // char argv_str2[] = "--gem5api";
    char* argv[] = {argv_str, /*argv_str2,*/ (char*)&gem5api_vector};

    pFunc gem5_main = (pFunc)lib.GetSymbol("gem5_main");

    int zephyr_ret = gem5_main(argc, argv);
    // zephyr_thread = new std::thread(gem5_main, argc, argv);
    DPRINTF(OpuCp, "gem5_main argc=%d, argv=%s, %d\n", argc, argv[0], zephyr_ret);
}

Port &
OpuCp::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "host_port")
        return hostPort;
    else if (if_name == "device_port")
        return devicePort;
    else if (if_name == "pio_port")
        return pioPort;
    else
        return ClockedObject::getPort(if_name, idx);
}

bool OpuCp::CPPort::recvTimingResp(PacketPtr pkt)
{
    cp->completeRequest(pkt);
    return true;
}

void OpuCp::CPPort::recvReqRetry()
{
    assert(outstandingPkts.size());

    DPRINTF(OpuCp, "Got a retry...\n");
    while (outstandingPkts.size() && sendTimingReq(outstandingPkts.front())) {
        DPRINTF(OpuCp, "Unblocked, sent blocked packet.\n");
        outstandingPkts.pop();
        // TODO: This should just signal the engine that the packet completed
        // engine should schedule tick as necessary. Need a test case
        if (!cp->memAccessTickEvent.scheduled()) {
            cp->schedule(cp->memAccessTickEvent, cp->nextCycle());
        }
    }
}

void OpuCp::CPPort::sendPacket(PacketPtr pkt) {
    if (isStalled() || !sendTimingReq(pkt)) {
        DPRINTF(OpuCp, "sendTiming failed in sendPacket(pkt->req->getVaddr()=0x%x)\n", (unsigned int)pkt->req->getVaddr());
        setStalled(pkt);
    }
}


void OpuCp::completeRequest(PacketPtr pkt, bool functional)
{
    const RequestPtr &req = pkt->req;
    /*
    // assert(req->getSize() == 4);        // only 4byte access

    // this address is no longer outstanding
    auto remove_addr = outstandingAddrs.find(req->getPaddr());
    assert(remove_addr != outstandingAddrs.end());
    outstandingAddrs.erase(remove_addr);
    */

    DPRINTF(OpuCp, "Completing %s at address vaddr:0x%x paddr:0x%x %s\n",
            pkt->isWrite() ? "write" : "read",
            req->getVaddr(), req->getPaddr(),
            pkt->isError() ? "error" : "success");

    // const uint32_t *pkt_data = pkt->getConstPtr<uint32_t>();

    if (pkt->isError()) {
        if (!functional || !suppressFuncWarnings) {
            panic("%s access failed at %#x\n",
                 pkt->isWrite() ? "Write" : "Read", req->getPaddr());
        }
    }

    if (pkt->isRead()) {
        pkt->writeData(curData + (req->getVaddr() - beginAddr));
        bytesRead += pkt->getSize();

        // set the addresses we just got as done
        for (int i = req->getVaddr() - beginAddr;
                i < req->getVaddr() - beginAddr + pkt->getSize(); i++) {
            readsDone[i] = true;
        }

        // mark readDone as only the contiguous region
        while (readDone < totalLength && readsDone[readDone]) {
            readDone++;
        }

        if (readDone >= totalLength) {
            DPRINTF(OpuCp, "done reading!!\n");
            needToRead = false;
            numOperations++;
            Tick total_time = curTick() - memAccessStartTime;
            DPRINTF(OpuCp, "Total time was: %llu\n", total_time);
            memAccessStats.push_back(MemAccessStats(total_time, memAccessLength));
            operationTimeTicks += total_time;
            running = false;
            // read_done_flag = true;
        }
    } else {
        assert(pkt->isWrite());

        writeDone += pkt->getSize();
        bytesWritten += pkt->getSize();
        if (!(writeDone < totalLength)) {
            // we are done!
            DPRINTF(OpuCp, "done writing, completely done!!!!\n");
            needToWrite = false;
            numOperations++;
            Tick total_time = curTick() - memAccessStartTime;
            DPRINTF(OpuCp, "Total time was: %llu\n", total_time);
            memAccessStats.push_back(MemAccessStats(total_time, memAccessLength));
            operationTimeTicks += total_time;
            running = false;
            // write_done_flag = true;
        } else {
            if (!memAccessTickEvent.scheduled()) {
                schedule(memAccessTickEvent, nextCycle());
            }
        }
    }

    // the packet will delete the data
    delete pkt;
}

CpRegIndex
decodeAddr(Addr paddr)
{
    CpRegIndex regNum;
    paddr &= ~mask(3);
    switch (paddr)
    {
      case 0x20:
        regNum = CP_ID;
        break;
      case 0x30:
        regNum = CP_VERSION;
        break;
      case 0x80:
        regNum = CP_RPTR;
        break;
      case 0x90:
        regNum = CP_WPTR;
        break;
      default:
        // A reserved register field.
        panic("Accessed reserved register field %#x.\n", paddr);
        break;
    }
    return regNum;
}

uint32_t
OpuCp::readReg(CpRegIndex reg)
{
    switch (reg) {
      case CP_ID:
        panic("Local APIC Processor Priority register unimplemented.\n");
        break;
      case CP_VERSION:
        regs[CP_VERSION] &= ~0x1ULL;
        break;
      default:
        break;
    }
    return regs[reg];
}

void
OpuCp::setReg(CpRegIndex reg, uint32_t val)
{
    uint32_t newVal = val;
    if (reg == CP_ID) {
        panic("APIC In-Service registers are unimplemented.\n");
    }
    switch (reg) {
      case CP_ID:
        newVal = val & 0xFF;
        break;
      case CP_VERSION:
        // The Local APIC Version register is read only.
        return;
      default:
        break;
    }
    regs[reg] = newVal;
    return;
}

AddrRangeList
OpuCp::getAddrRanges() const
{
    AddrRangeList ranges;
    DPRINTF(OpuCp, "CP registering addr range at %#x size %#x\n",
            pioAddr, pioSize);

    ranges.push_back(RangeSize(pioAddr, pioSize));
    return ranges;
}


Tick
OpuCp::read(PacketPtr pkt)
{
    assert(pkt->getAddr() >= pioAddr);
    assert(pkt->getAddr() < pioAddr + pioSize);

    Addr offset = pkt->getAddr() - pioAddr;
    pkt->allocate();

    // CP_NUM_REGS is not context regs
    if (offset < CP_NUM_REGS) {
        // Make sure we're at least only accessing one register.
        if ((offset & ~mask(3)) != ((offset + pkt->getSize()) & ~mask(3)))
            panic("Accessed more than one register at a time in the CP!\n");


        CpRegIndex reg = decodeAddr(offset);
        uint32_t val = htole(readReg(reg));

        DPRINTF(OpuCp, " read CP register %d at offset %#x size=%d\n", reg,  offset, pkt->getSize());

        pkt->setData(((uint8_t *)&val) + (offset & mask(3)));
    } else {
        /* TODO
        assert(offset + pkt->getSize() < sizeof(HsaQueueEntry));
        char *curCtxPtr = (char*)&curContext;

        memcpy(pkt->getPtr<const void*>(), curCtxPtr + offset, pkt->getSize());
        */
    }


    pkt->makeAtomicResponse();

    return pioDelay;
}

Tick
OpuCp::write(PacketPtr pkt)
{
    assert(pkt->getAddr() >= pioAddr);
    assert(pkt->getAddr() < pioAddr + pioSize);
    Addr offset = pkt->getAddr() - pioAddr;

    if (offset < CP_NUM_REGS) {
        // Make sure we're at least only accessing one register.
        if ((offset & ~mask(3)) != ((offset + pkt->getSize()) & ~mask(3)))
            panic("Accessed more than one register at a time in the APIC!\n");
        CpRegIndex reg = decodeAddr(offset);
        uint32_t val = regs[reg];

        pkt->writeData(((uint8_t *)&val) + (offset & mask(3)));

        DPRINTF(OpuCp,
            "Writing CP register %d at offset %#x as %#x.\n",
            reg, offset, letoh(val));
        setReg(reg, letoh(val));
    } else {
        // context regs
    }

    pkt->makeAtomicResponse();
    return pioDelay;
}


#if 0
void
OpuCp::memAccess( bool cmd_read, uint32_t data, bool uncacheable, Addr paddr, bool functional)
{
    // we should never tick if we are waiting for a retry
    assert(!retryPkt);

    // create a new request
    Request::Flags flags;

    // Tick t = curTick();
    // DPRINTF(OpuCp, "Call in memAccess is curTick %ld\n", t);

    // use the tester id as offset within the block for false sharing
    // paddr = blockAlign(paddr);

    if (uncacheable) {
        flags.set(Request::UNCACHEABLE);
    }

    bool do_functional = functional && !uncacheable;
    RequestPtr req = std::make_shared<Request>(paddr, 4, flags, masterId); // it is 4Byte access
    req->setContext(id);

    outstandingAddrs.insert(paddr);

    // sanity check
    panic_if(outstandingAddrs.size() > 100,
             "Tester %s has more than 100 outstanding requests\n", name());

    PacketPtr pkt = nullptr;
    // uint8_t *pkt_data = new uint8_t[4];
    uint32_t *pkt_data = new uint32_t[1];

    if (cmd_read) {
        // start by ensuring there is a reference value if we have not
        // seen this address before
        uint32_t M5_VAR_USED ref_data = 0;
        auto ref = referenceData.find(req->getPaddr());
        if (ref == referenceData.end()) {
            referenceData[req->getPaddr()] = 0;
        } else {
            ref_data = ref->second;
        }

        DPRINTF(OpuCp,
                "Initiating %sread at addr %x (blk %x) expecting %x\n",
                do_functional ? "functional " : "", req->getPaddr(),
                blockAlign(req->getPaddr()), ref_data);

        pkt = new Packet(req, MemCmd::ReadReq);
        pkt->dataDynamic(pkt_data);
    } else {
        DPRINTF(OpuCp, "Initiating %swrite at addr %x (blk %x) value %x\n",
                do_functional ? "functional " : "", req->getPaddr(),
                blockAlign(req->getPaddr()), data);

        pkt = new Packet(req, MemCmd::WriteReq);
        pkt->dataDynamic(pkt_data);
        pkt_data[0] = data;
    }

    // there is no point in ticking if we are waiting for a retry
    //bool keep_ticking = true;
    if (do_functional) {
        pkt->setSuppressFuncError();
        port.sendFunctional(pkt);
        completeRequest(pkt, true);
    } else {
        /*keep_ticking = */sendPkt(pkt);
    }
}
#endif

void OpuCp::regStats() {

    ClockedObject::regStats();
    using namespace Stats;

    numOperations
        .name(name() + ".numOperations")
        .desc("Number of copy/memset operations")
        ;
    bytesRead
        .name(name() + ".opBytesRead")
        .desc("Number of copy bytes read")
        ;
    bytesWritten
        .name(name() + ".opBytesWritten")
        .desc("Number of copy/memset bytes written")
        ;
    operationTimeTicks
        .name(name() + ".opTimeTicks")
        .desc("Total time spent in copy/memset operations")
        ;

    if (hostDTB) { hostDTB->regStats(); }
    if (deviceDTB) { deviceDTB->regStats(); }
    if (readDTB) { readDTB->regStats(); }
    if (writeDTB) { writeDTB->regStats(); }
}

void
OpuCp::init()
{
    panic_if(!hostPort.isConnected(),
            "Host port not connected to anything!");
    panic_if(!devicePort.isConnected(),
            "device port not connected to anything!");
    panic_if(!pioPort.isConnected(),
            "Pio port of %s not connected to anything!", name());

    pioPort.sendRangeChange();
}
/*
OpuCp *
OpuCpParams::create() const
{
    return new OpuCp(*this);
}
*/
}
