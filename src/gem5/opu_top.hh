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

#ifndef __OPU_TOP_HH__
#define __OPU_TOP_HH__

#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "stream_manager.h"
#include "base/callback.hh"
#include "debug/OpuTop.hh"
#include "debug/OpuTopPageTable.hh"
#include "opusim_base.hh"
#include "opu_core.hh"
#include "gpu/copy_engine.hh"
#include "gpu/command_processor.hh"
#include "gpu/shader_mmu.hh"
#include "params/OpuTop.hh"
#include "params/OpuSimComponentWrapper.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include "sim/process.hh"
#include "sim/system.hh"

namespace gem5 {
/**
 * A wrapper class to manage the clocking of GPGPU-Sim-side components.
 * The OpuTop must contain one of these wrappers for each clocked component or
 * else GPGPU-Sim simulation progress will stall. Currently, there are four
 * GPGPU-Sim components that are separately cycled: the shader cores, the
 * interconnect, the GPU L2 cache and the DRAM.
 *
 * TODO: Eventually, the L2 and DRAM events should be eliminated by migrating
 * all GPU parameter and local memory accesses over to gem5-gpu.
 */
class OpuSimComponentWrapper : public ClockedObject
{
  private:
    OpuSimBase *theGPU;
    typedef void (OpuSimBase::*CycleFunc)();
    CycleFunc startCycleFunction;
    CycleFunc endCycleFunction;

  public:
    OpuSimComponentWrapper(const OpuSimComponentWrapperParams &p) :
        ClockedObject(p), theGPU(NULL), startCycleFunction(NULL),
        endCycleFunction(NULL), componentCycleStartEvent(this),
        // End cycle events must happen after all other components are cycled
        componentCycleEndEvent(this, false, Event::Progress_Event_Pri) {}

    void setGPU(OpuSimBase *_gpu) {
        assert(!theGPU);
        theGPU = _gpu;
    }

    void setStartCycleFunction(CycleFunc _cycle_func) {
        assert(!startCycleFunction);
        startCycleFunction = _cycle_func;
    }

    void setEndCycleFunction(CycleFunc _cycle_func) {
        assert(!endCycleFunction);
        endCycleFunction = _cycle_func;
    }

    void scheduleEvent(Tick ticks_in_future) {
        Tick start_time;
        if (ticks_in_future < clockPeriod()) {
            start_time = nextCycle();
        } else {
            start_time = clockEdge(ticksToCycles(ticks_in_future));
        }

        assert(startCycleFunction);
        assert(!componentCycleStartEvent.scheduled());
        schedule(componentCycleStartEvent, start_time);

        if (endCycleFunction) {
            assert(!componentCycleEndEvent.scheduled());
            schedule(componentCycleEndEvent, start_time);
        }
    }

  protected:

    void componentCycleStart() {
        assert(startCycleFunction);

        if (theGPU->active()) {
            (theGPU->*startCycleFunction)();
        }

        if (theGPU->active()) {
            // Reschedule the start cycle event
            schedule(componentCycleStartEvent, nextCycle());
        }
    }

    void componentCycleEnd() {
        assert(endCycleFunction);

        if (theGPU->active()) {
            (theGPU->*endCycleFunction)();
        }

        if (theGPU->active()) {
            // Reschedule the end cycle event
            schedule(componentCycleEndEvent, nextCycle());
        }
    }

    EventWrapper<OpuSimComponentWrapper, &OpuSimComponentWrapper::componentCycleStart>
                                           componentCycleStartEvent;
    EventWrapper<OpuSimComponentWrapper, &OpuSimComponentWrapper::componentCycleEnd>
                                           componentCycleEndEvent;
};

/**
 *  Main wrapper class for GPGPU-Sim
 *
 *  All global and const accesses from GPGPU-Sim are routed through this class.
 *  This class also holds pointers to all of the CUDA cores and the copy engine.
 *  Statistics for kernel times are also kept in this class.
 *
 *  Currently this class only supports a single GPU device and does not support
 *  concurrent kernels.
 */
class OpuTop : public ClockedObject
{
  private:
    static std::vector<OpuTop*> gpuArray;
    // static Tick os_tick;

  public:
    /**
     *  Only to be used in GPU system calls (gpu_syscalls) as a way to access
     *  the CUDA-enabled GPUs.
     */
    static OpuTop *getOpuTop(unsigned id) {
        if (id >= gpuArray.size()) {
            panic("CUDA GPU ID not found: %u. Only %u GPUs registered!\n", id, gpuArray.size());
        }
        static bool initialized = false;
        if (not initialized) {
            // event_queue = new EventQueue("OpuTopEventQueue");
            curEventQueue(gpuArray[id]->eventQueue());
            initialized = true;
            // event_queue->setCurTick(0);
        }
        return gpuArray[id];
    }

    static unsigned getNumOpuDevices() {
        return gpuArray.size();
    }

    static unsigned registerOpuDevice(OpuTop *gpu) {
        unsigned new_id = getNumOpuDevices();
        gpuArray.push_back(gpu);
        return new_id;
    }

    gpgpu_context * getGPGPUCtx() {
        return gpgpu_ctx;
    }

    stream_manager * getStreamManager () {
        return streamManager;
    };

    struct OpuDeviceProperties
    {
        char   name[256];                 // ASCII string identifying device
        size_t totalGlobalMem;            // Global memory available on device in bytes
        size_t sharedMemPerBlock;         // Shared memory available per block in bytes
        int    regsPerBlock;              // 32-bit registers available per block
        int    warpSize;                  // Warp size in threads
        size_t memPitch;                  // Maximum pitch in bytes allowed by memory copies
        int    maxThreadsPerBlock;        // Maximum number of threads per block
        int    maxThreadsDim[3];          // Maximum size of each dimension of a block
        int    maxGridSize[3];            // Maximum size of each dimension of a grid
        int    clockRate;                 // Clock frequency in kilohertz
        size_t totalConstMem;             // Constant memory available on device in bytes
        int    major;                     // Major compute capability
        int    minor;                     // Minor compute capability
        size_t textureAlignment;          // Alignment requirement for textures
        int    deviceOverlap;             // Device can concurrently copy memory and execute a kernel
        int    multiProcessorCount;       // Number of multiprocessors on device
        int    kernelExecTimeoutEnabled;  // Specified whether there is a run time limit on kernels
        int    integrated;                // Device is integrated as opposed to discrete
        int    canMapHostMemory;          // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        int    computeMode;               // Compute mode (See ::cudaComputeMode)
        int    maxTexture1D;              // Maximum 1D texture size
        int    maxTexture2D[2];           // Maximum 2D texture dimensions
        int    maxTexture3D[3];           // Maximum 3D texture dimensions
        int    maxTexture2DArray[3];      // Maximum 2D texture array dimensions
        size_t surfaceAlignment;          // Alignment requirements for surfaces
        int    concurrentKernels;         // Device can possibly execute multiple kernels concurrently
        int    ECCEnabled;                // Device has ECC support enabled
        int    pciBusID;                  // PCI bus ID of the device
        int    pciDeviceID;               // PCI device ID of the device
        int    __cudaReserved[22];
    };

  protected:
    typedef OpuTopParams Params;

    /**
     *  Helper class for both Stream and GPU tick events
     */
    class TickEvent : public Event
    {
        friend class OpuTop;

      private:
        OpuTop *cpu;

      public:
        TickEvent(OpuTop *c) : Event(CPU_Tick_Pri), cpu(c) {}
        void process() {
            cpu->streamTick();
        }
        virtual const char *description() const { return "OpuTop tick"; }
    };

    class FinishKernelEvent : public Event
    {
        friend class OpuTop;

    private:
        OpuTop *gpu;
        int gridId;
    public:
        FinishKernelEvent(OpuTop *_gpu, int grid_id) :
            gpu(_gpu), gridId(grid_id)
        {
            setFlags(Event::AutoDelete);
        }
        void process() {
            gpu->processFinishKernelEvent(gridId);
        }
    };

    const OpuTopParams *_params;
    const Params * params() const { return dynamic_cast<const Params *>(_params); }

    /// Tick for when the stream manager needs execute
    TickEvent streamTickEvent;

  private:
    // The CUDA device ID for this GPU
    unsigned cudaDeviceID;

    // Clock domain for the GPU: Used for changing frequency
    SrcClockDomain *clkDomain;

    // Wrappers to cycle components in GPGPU-Sim
    OpuSimComponentWrapper &coresWrapper;
    // OpuSimComponentWrapper &icntWrapper;
    // OpuSimComponentWrapper &l2Wrapper;
    // OpuSimComponentWrapper &dramWrapper;

    /// Callback for the stream manager tick
    void streamTick();

    /// Pointer to the copy engine for this device
    GPUCopyEngine *copyEngine;

    /// Pointer to the copy engine for this device
    CommandProcessor *commandProcessor;

    /// Used to register this SPA with the system
    System *system;

    /// Number of threads in each warp, also number of lanes per CUDA core
    int warpSize;

    /// Are we restoring from a checkpoint?
    bool restoring;

    int sharedMemDelay;
    std::string gpgpusimConfigPath;
    Tick launchDelay;
    Tick returnDelay;

    /// If true there is a kernel currently executing
    /// NOTE: Jason doesn't think we need this
    bool running;

    /// True if the running thread is currently blocked and needs to be activated
    bool unblockNeeded;

    /// Pointer to ruby system used to clear the Ruby stats
    /// NOTE: I think there is a more right way to do this
    ruby::RubySystem *ruby;

    /// Holds all of the CUDA cores in this GPU
    std::vector<OpuCore*> opuCores;

    /// The thread context, stream and thread ID currently running on the SPA
    ThreadContext *runningTC;
    struct CUstream_st *runningStream;
    int runningTID;
    Addr runningPTBase;
    void beginStreamOperation(struct CUstream_st *_stream) {
        // We currently do not support multiple concurrent streams
        if (runningStream || runningTC) {
            panic("Already a stream operation running (only support one at a time)!");
        }
        // NOTE: This may cause a race: The runningTC may have changed (i.e.
        // the thread was migrated) between when the thread queued the stream
        // operation and when that operation starts executing here. By reading
        // CR3 here, we could use this to double check that the correct thread
        // is running. On the other hand, we could move the CR3 read into the
        // operation queuing code to avoid the race, but we would not be able
        // to detect of the thread had migrated since it queued the operation.
        runningStream = _stream;
        runningTC = runningStream->getThreadContext();
        runningTID = runningTC->threadId();
#if THE_ISA == X86_ISA
        runningPTBase = runningTC->readMiscRegNoEffect(X86ISA::MISCREG_CR3);
#else
        // TODO: ARM ISA should use the TTBCR for user space (which appears
        // to be called the TTBR1 register). Further investigation required.
        warn_once("ISA's pagetable base register handling needs to be set up");
#endif
    }
    void endStreamOperation() {
        runningStream = NULL;
        runningTC = NULL;
        runningTID = -1;
        runningPTBase = 0;
    }

    /// For statistics
    std::vector<Tick> kernelTimes;
    Tick clearTick;
    bool dumpKernelStats;

    /// Pointers to GPGPU-Sim objects
    OpuSimBase *theGPU;
    stream_manager *streamManager;
    gpgpu_context *gpgpu_ctx;

    /// Flag to make sure we don't schedule twice in the same tick
    bool streamScheduled;

    /// Number of ticks to delay for each stream operation
    /// This is a function of the driver overheads
    int streamDelay;

    /// For GPU syscalls
    /// This is what is required to save and restore on checkpoints
    std::map<const void*,function_info*> m_kernel_lookup; // unique id (CUDA app function address) => kernel entry point
    uint64_t instBaseVaddr;
    bool instBaseVaddrSet;
    Addr localBaseVaddr;

    class GPUPageTable
    {
      private:
        std::map<Addr, Addr> pageMap;

      public:
        GPUPageTable() {};

        Addr addrToPage(Addr addr);
        void insert(Addr vaddr, Addr paddr) {
            if (pageMap.find(vaddr) == pageMap.end()) {
                pageMap[vaddr] = paddr;
            } else {
                assert(paddr == pageMap[vaddr]);
            }
        }
        bool lookup(Addr vaddr, Addr& paddr) {
            Addr page_vaddr = addrToPage(vaddr);
            Addr offset = vaddr - page_vaddr;
            if (pageMap.find(page_vaddr) != pageMap.end()) {
                paddr = pageMap[page_vaddr] + offset;
                return true;
            }
            return false;
        }
        /// For checkpointing
        void serialize(CheckpointOut &cp) const;
        void unserialize(CheckpointIn &cp);
    };
    GPUPageTable pageTable;
    bool manageGPUMemory;
    bool accessHostPageTable;
    AddrRange gpuMemoryRange;
    Addr physicalGPUBrkAddr;
    Addr virtualGPUBrkAddr;
    Addr cpMemoryBaseVaddr;
    Addr cpMemoryBaseSize;
    std::map<Addr,size_t> allocatedGPUMemory;

    ShaderMMU *shaderMMU;

    OpuDeviceProperties deviceProperties;

  public:
    /// Constructor
    OpuTop(const OpuTopParams &p);

    /// For checkpointing
    virtual void serialize(CheckpointOut &cp) const;
    virtual void unserialize(CheckpointIn &cp);

    /// Called after constructor, but before any real simulation
    virtual void startup();

    /// Register devices callbacks
    void registerOpuCore(OpuCore *sc);
    void registerCopyEngine(GPUCopyEngine *ce);
    void registerCommandProcessor(CommandProcessor *cp);

    /// Getter for whether we are using Ruby or GPGPU-Sim memory modeling
    OpuDeviceProperties *getDeviceProperties() { return &deviceProperties; }
    unsigned getMaxThreadsPerMultiprocessor() {
        if (deviceProperties.major == 2) {
            warn("Returning threads per multiprocessor from compute capability 2.x\n");
            return 1536;
        }
        panic("Have not configured threads per multiprocessor!\n");
        return 0;
    }
    int getSharedMemDelay() { return sharedMemDelay; }
    const char* getConfigPath() { return gpgpusimConfigPath.c_str(); }
    ruby::RubySystem* getRubySystem() { return ruby; }
    OpuSimBase* getTheGPU() { return theGPU; }

    /// Called at the beginning of each kernel launch to start the statistics
    void beginRunning(Tick stream_queued_time, struct CUstream_st *_stream);

    /**
     * Marks the kernel as complete and signals the stream manager
     */
    void processFinishKernelEvent(int grid_id);

    /**
     * Called from GPGPU-Sim when the kernel completes on all shaders
     */
    void finishKernel(int grid_id);

    void handleFinishPageFault(ThreadContext *tc)
        { shaderMMU->handleFinishPageFault(tc); }

    ShaderMMU *getMMU() { return shaderMMU; }

    /// Schedules the stream manager to be checked in 'ticks' ticks from now
    void scheduleStreamEvent();

    /// Reset statistics for the SPA and for all of Ruby
    void clearStats();

    /// Returns CUDA core with id coreId
    OpuCore *getOpuCore(int coreId);

    /// Returns size of warp (same for all CUDA cores)
    int getWarpSize() { return warpSize; }

    /// Callback for GPGPU-Sim to get the current simulation time
    Tick getCurTick(){ return curTick(); }

    /// Used to print stats at the end of simulation
    void gpuPrintStats(std::ostream& out);


    /// Begins a timing memory copy from src to dst
    void memcpy(void *src, void *dst, size_t count, struct CUstream_st *stream, stream_operation_type type);

    /// Begins a timing memory copy from src to/from the symbol+offset
    void memcpy_to_symbol(const char *hostVar, const void *src, size_t count, size_t offset, struct CUstream_st *stream);
    void memcpy_from_symbol(void *dst, const char *hostVar, size_t count, size_t offset, struct CUstream_st *stream);

    /// Begins a timing memory set of value to dst
    void memset(Addr dst, int value, size_t count, struct CUstream_st *stream);

    /// Called by the copy engine when a memcpy or memset is complete
    void finishCopyOperation();

    /// Called from shader TLB to be used for TLB lookups
    /// TODO: Move the thread context handling to GPU context when we get there
    ThreadContext *getThreadContext() { return runningTC; }
    Addr getRunningPTBase() { return runningPTBase; }
    void checkUpdateThreadContext(ThreadContext *tc) {
        if (!runningTC) {
            // The GPU isn't running anything, so it won't try to access the
            // thread context for anything (e.g. address translations). Hence,
            // it is safe to ignore this check/update process
            return;
        }
        if (tc != runningTC) {
#if THE_ISA == X86_ISA
            Addr pagetable_base = tc->readMiscRegNoEffect(X86ISA::MISCREG_CR3);
#else
            warn_once("ISA's pagetable base needs to be read and checked!");
            Addr pagetable_base = 0;
#endif
            warn("Thread migrated! Old tc: %p, PT: %p, New tc: %p, PT: %p\n",
                 runningTC, runningPTBase, tc, pagetable_base);
            if (pagetable_base == runningPTBase) {
                // No problem, just change migrate the runningTC
                DPRINTF(OpuTop, "Updating the thread context\n");
                runningTC = tc;
                runningTID = runningTC->threadId();
            } else {
                panic("New pagetable address doesn't match old!\n");
            }
        }
        // NOTE: If we can get away with live updating the thread context
        // pointer while the GPU is executing, we need to make sure that the
        // ShaderMMU and ShaderTLBs use the latest runningTC (i.e. we may need
        // to dynamically update the tc of in-flight translations, and squash
        // those that are in page-walks). This possibility seems unlikely.
    }

    /// Used when blocking and signaling threads
    std::map<ThreadContext*, Addr> blockedThreads;
    bool needsToBlock();
    void blockThread(ThreadContext *tc, Addr signal_ptr);
    void signalThread(ThreadContext *tc, Addr signal_ptr);
    void unblockThread(ThreadContext *tc);

    /// From gpu syscalls (used to be CUctx_st)
    function_info *get_kernel(const char *hostFun);
    void setInstBaseVaddr(uint64_t addr);
    uint64_t getInstBaseVaddr();
    void setLocalBaseVaddr(uint64_t addr);
    uint64_t getLocalBaseVaddr();

    /// For handling GPU memory mapping table
    GPUPageTable* getGPUPageTable() { return &pageTable; };
    void registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size);
    void registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size);
    bool isManagingGPUMemory() { return manageGPUMemory; }
    bool isAccessingHostPagetable() { return accessHostPageTable; }
    Addr allocateGPUMemory(size_t size);

    /// Statistics for this GPU
    Stats::Scalar numKernelsStarted;
    Stats::Scalar numKernelsCompleted;
    void regStats();

    bool is_active() { return theGPU->active(); }

    void callback();

    OutputStream* statsFile;
};

/**
 *  Helper class to print out statistics at the end of simulation
 */
/*
class GPUExitCallback : public Callback
{
  private:
    std::string stats_filename;
    OpuTop *gpu;

  public:
    virtual ~GPUExitCallback() {}

    GPUExitCallback(OpuTop *_gpu, const std::string& _stats_filename)
    {
        stats_filename = _stats_filename;
        gpu = _gpu;
    }

    virtual void process();
};
*/
}

#endif

