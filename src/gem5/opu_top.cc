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
#include <sstream>
#include <string>

#include "api/gpu_syscall_helper.hh"
//#include "arch/utility.hh"
//#include "arch/vtophys.hh"
#include "arch/x86/regs/misc.hh"
#include "base/chunk_generator.hh"
#include "base/statistics.hh"
#include "base/intmath.hh"
#include "cpu/thread_context.hh"
// #include "cuda-sim/cuda-sim.h"
// #include "cuda-sim/ptx-stats.h"
#include "debug/OpuTop.hh"
#include "debug/OpuTopAccess.hh"
#include "debug/OpuTopPageTable.hh"
#include "debug/OpuTopTick.hh"
// #include "gpgpusim_entrypoint.h"
// #include "../libcuda_sim/gpgpu_context.h"
#include "opu_context.hh"
#include "opu_top.hh"
#include "opusim_base.hh"
#include "mem/ruby/system/RubySystem.hh"
#include "params/OpuTop.hh"
#include "params/OpuSimComponentWrapper.hh"
#include "sim/full_system.hh"

using namespace std;

// FIXME
// int no_of_ptx = 0;
// char *ptx_line_stats_filename = "ptx_line_stats.rpt";
OpuSimBase* g_the_opu;

namespace gem5 {

vector<OpuTop*> OpuTop::gpuArray;

// From GPU syscalls
// void registerFatBinaryTop(GPUSyscallHelper *helper, Addr sim_fatCubin, size_t sim_binSize);
// unsigned int registerFatBinaryBottom(GPUSyscallHelper *helper, Addr sim_alloc_ptr);
// void register_var(Addr sim_deviceAddress, const char* deviceName, int sim_size, int sim_constant, int sim_global, int sim_ext, Addr sim_hostVar);

static std::mutex tick_mutex;

OpuTop::OpuTop(const OpuTopParams &p) :
    ClockedObject(p), _params(&p), streamTickEvent(this),
    clkDomain((SrcClockDomain*)p.clk_domain),
    coresWrapper(*p.cores_wrapper),/* icntWrapper(*p.icnt_wrapper),
    l2Wrapper(*p.l2_wrapper), dramWrapper(*p.dram_wrapper),*/
    system(p.sys), warpSize(p.warp_size), sharedMemDelay(p.shared_mem_delay),
    gpgpusimConfigPath(p.config_path), unblockNeeded(false), ruby(p.ruby),
    runningTC(NULL), runningStream(NULL), runningTID(-1), runningPTBase(0),
    clearTick(0), dumpKernelStats(p.dump_kernel_stats), pageTable(),
    manageGPUMemory(p.manage_gpu_memory),
    accessHostPageTable(p.access_host_pagetable),
    gpuMemoryRange(p.gpu_memory_range), shaderMMU(p.opu_mmu)
{
    // Register this device as a CUDA-enabled GPU
    cudaDeviceID = registerOpuDevice(this);
    if (cudaDeviceID >= 1) {
        // TODO: Remove this when multiple GPUs can exist in system
        panic("GPGPU-Sim is not currently able to simulate more than 1 CUDA-enabled GPU\n");
    }

    streamDelay = 1;

    running = false;

    streamScheduled = false;

    restoring = false;

    launchDelay = p.kernel_launch_delay * SimClock::Frequency;
    returnDelay = p.kernel_return_delay * SimClock::Frequency;

    // GPU memory handling
    instBaseVaddr = 0;
    instBaseVaddrSet = false;
    localBaseVaddr = 0;

    // Reserve the 0 virtual page for NULL pointers
    virtualGPUBrkAddr = TheISA::PageBytes;
    physicalGPUBrkAddr = gpuMemoryRange.start();

    // CP memory scratch
    cpMemoryBaseVaddr = virtualGPUBrkAddr;
    cpMemoryBaseSize = TheISA::PageBytes * 10;  // reserve 10 page for cp scratch

    // Initiaalize default gpgpu_context
    opu_ctx = new OpuContext();

    // Initialize GPGPU-Sim
    theOpuUsim = opu_ctx->gem5_opu_sim_init(&streamManager, this, getConfigPath());
    // FIXME
    theOpuUsim->init();

    g_the_opu = theOpuUsim;

    // Set up the component wrappers in order to cycle the GPGPU-Sim
    // shader cores, interconnect, L2 cache and DRAM
    // TODO: Eventually, we want to remove the need for the GPGPU-Sim L2 cache
    // and DRAM. Currently, these are necessary to handle parameter memory
    // accesses.
    coresWrapper.setGPU(theOpuUsim);
    coresWrapper.setStartCycleFunction(&OpuSimBase::core_cycle_start);
    coresWrapper.setEndCycleFunction(&OpuSimBase::core_cycle_end);
    // icntWrapper.setGPU(theOpuUsim);
    // icntWrapper.setStartCycleFunction(&OpuSimBase::icnt_cycle_start);
    // icntWrapper.setEndCycleFunction(&OpuSimBase::icnt_cycle_end);
    // l2Wrapper.setGPU(theOpuUsim);
    // l2Wrapper.setStartCycleFunction(&OpuSimBase::l2_cycle);
    // dramWrapper.setGPU(theOpuUsim);
    // dramWrapper.setStartCycleFunction(&OpuSimBase::dram_cycle);

    // Setup the device properties for this GPU
    // snprintf(deviceProperties.name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    deviceProperties.major = 2;
    deviceProperties.minor = 0;
    deviceProperties.totalGlobalMem = gpuMemoryRange.size();
    deviceProperties.memPitch = 0;
    deviceProperties.maxThreadsPerBlock = 1024;
    deviceProperties.maxThreadsDim[0] = 1024;
    deviceProperties.maxThreadsDim[1] = 1024;
    deviceProperties.maxThreadsDim[2] = 64;
    deviceProperties.maxGridSize[0] = 0x40000000;
    deviceProperties.maxGridSize[1] = 0x40000000;
    deviceProperties.maxGridSize[2] = 0x40000000;
    deviceProperties.totalConstMem = gpuMemoryRange.size();
    deviceProperties.textureAlignment = 0;
    deviceProperties.multiProcessorCount = opuCores.size();
    deviceProperties.sharedMemPerBlock = theOpuUsim->shared_mem_size();
    deviceProperties.regsPerBlock = theOpuUsim->num_registers_per_core();
    deviceProperties.warpSize = theOpuUsim->wrp_size();
    deviceProperties.clockRate = theOpuUsim->shader_clock();
#if (CUDART_VERSION >= 2010)
    // FIXME deviceProperties.multiProcessorCount = theOpuUsim->get_config().num_shader();
#endif
    OutputStream *os = simout.find(p.stats_filename);
    if (!os) {
        statsFile = simout.create(p.stats_filename);
    }

    assert(statsFile);


    // Print gpu configuration and stats at exit
    // GPUExitCallback* gpuExitCB = new GPUExitCallback(this, p.stats_filename);
    // registerExitCallback(gpuExitCB);
    registerExitCallback([this] () { callback(); });

}

void OpuTop::serialize(CheckpointOut &cp) const
{
    DPRINTF(OpuTop, "Serializing\n");
    if (running) {
        panic("Checkpointing during GPU execution not supported\n");
    }
    pageTable.serialize(cp);
}

void OpuTop::unserialize(CheckpointIn &cp)
{
    DPRINTF(OpuTop, "UNserializing\n");

    pageTable.unserialize(cp);
}

void OpuTop::startup()
{
    // Initialize CUDA cores
    vector<OpuCore*>::iterator iter;
    for (iter = opuCores.begin(); iter != opuCores.end(); ++iter) {
        (*iter)->initialize();
    }

    if (!restoring) {
        return;
    }

    if (runningTID >= 0) {
        runningTC = system->threads[runningTID];
        assert(runningTC);
    }
    // FIXME don't support checkpoint
#if 0
    // Setting everything up again!
    std::vector<_FatBinary>::iterator binaries;
    for (binaries = fatBinaries.begin(); binaries != fatBinaries.end(); ++binaries) {
        _FatBinary bin = *binaries;
        GPUSyscallHelper helper(system->getThreadContext(bin.tid));
        assert(helper.getThreadContext());
        registerFatBinaryTop(&helper, bin.sim_fatCubin, bin.sim_binSize);
        registerFatBinaryBottom(&helper, bin.sim_alloc_ptr);

        std::map<const void*, string>::iterator functions;
        for (functions = bin.funcMap.begin(); functions != bin.funcMap.end(); ++functions) {
            const char *host_fun = (const char*)functions->first;
            const char *device_fun = functions->second.c_str();
            register_function(bin.handle, host_fun, device_fun);
        }
    }

    std::vector<_CudaVar>::iterator variables;
    for (variables = cudaVars.begin(); variables != cudaVars.end(); ++variables) {
        _CudaVar var = *variables;
        register_var(var.sim_deviceAddress, var.deviceName.c_str(), var.sim_size, var.sim_constant, var.sim_global, var.sim_ext, var.sim_hostVar);
    }
#endif
}

void OpuTop::clearStats()
{
    ruby->resetStats();
    clearTick = curTick();
}

void OpuTop::registerOpuCore(OpuCore *sc)
{
    opuCores.push_back(sc);

    // Update the multiprocessor count
    deviceProperties.multiProcessorCount = opuCores.size();
}

void OpuTop::registerCopyEngine(OpuDma *ce)
{
    copyEngine = ce;
}

void OpuTop::registerCp(OpuCp *cp)
{
    commandProcessor = cp;
}

void OpuTop::streamTick() {
    std::lock_guard<std::mutex> lock(tick_mutex);
    DPRINTF(OpuTopTick, "Stream Tick\n");

    streamScheduled = false;
    // FIXME
#if 0
    // launch operation on device if one is pending and can be run
    stream_operation op = streamManager->front();
    bool kickoff = op.do_operation(theOpuUsim);

    if (!kickoff) {
        //cancel operation
        //if( op.is_kernel() ) {
        //    unsigned grid_uid = op.get_kernel()->get_uid();
        //    m_grid_id_to_stream.erase(grid_uid);
        //}
        op.get_stream()->cancel_front();
    }

    if (!kickoff || streamManager->ready()) {
        schedule(streamTickEvent, curTick() + streamDelay);
        streamScheduled = true;
    }
#endif
}

void OpuTop::scheduleStreamEvent() {
    std::lock_guard<std::mutex> lock(tick_mutex);
    if (streamScheduled) {
        DPRINTF(OpuTopTick, "Already scheduled a tick, ignoring\n");
        return;
    }

    schedule(streamTickEvent, nextCycle());
    streamScheduled = true;
}

void OpuTop::beginRunning(Tick stream_queued_time, struct Stream_st *_stream)
{
    beginStreamOperation(_stream);

    DPRINTF(OpuTop, "Beginning kernel execution at %llu\n", curTick());
    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        Stats::dump();
        Stats::reset();
    }
    numKernelsStarted++;
    if (running) {
        panic("Should not already be running if we are starting\n");
    }
    running = true;

    Tick delay = clockPeriod();
    if ((stream_queued_time + launchDelay) > curTick()) {
        // Delay launch to the end of the launch delay
        delay = (stream_queued_time + launchDelay) - curTick();
    }

    coresWrapper.scheduleEvent(delay);
    // icntWrapper.scheduleEvent(delay);
    // l2Wrapper.scheduleEvent(delay);
    // dramWrapper.scheduleEvent(delay);
}

void OpuTop::finishKernel(int grid_id)
{
    numKernelsCompleted++;
    FinishKernelEvent *e = new FinishKernelEvent(this, grid_id);
    schedule(e, curTick() + returnDelay);
}

void OpuTop::processFinishKernelEvent(int grid_id)
{
    DPRINTF(OpuTop, "GPU finished a kernel id %d\n", grid_id);

    streamManager->register_finished_kernel(grid_id);

    kernelTimes.push_back(curTick());
    if (dumpKernelStats) {
        Stats::dump();
        Stats::reset();
    }

    if (unblockNeeded && streamManager->empty()) {
        DPRINTF(OpuTop, "Stream manager is empty, unblocking\n");
        unblockThread(runningTC);
    }

    scheduleStreamEvent();

    running = false;

    endStreamOperation();
}

OpuCore *OpuTop::getOpuCore(int coreId)
{
    assert(coreId < opuCores.size());
    return opuCores[coreId];
}

/*
OpuTop *OpuTopParams::create() const {
    return new OpuTop(*this);
}
*/

void OpuTop::gpuPrintStats(std::ostream& out) {
    // Print kernel statistics
    Tick total_kernel_ticks = 0;
    Tick last_kernel_time = 0;
    bool kernel_active = false;
    vector<Tick>::iterator it;

    out << "spa frequency: " << frequency()/1000000000.0 << " GHz\n";
    out << "spa period: " << clockPeriod() << " ticks\n";
    out << "kernel times (ticks):\n";
    out << "start, end, start, end, ..., exit\n";
    for (it = kernelTimes.begin(); it < kernelTimes.end(); it++) {
        out << *it << ", ";
        if (kernel_active) {
            total_kernel_ticks += (*it - last_kernel_time);
            kernel_active = false;
        } else {
            last_kernel_time = *it;
            kernel_active = true;
        }
    }
    out << curTick() << "\n";

    // Print Shader CTA statistics
    out << "\nshader CTA times (ticks):\n";
    out << "shader, CTA ID, start, end, start, end, ..., exit\n";
    std::vector<OpuCore*>::iterator cores;
#if 0
    for (cores = opuCores.begin(); cores != opuCores.end(); cores++) {
        (*cores)->printCTAStats(out);
    }
#endif
    out << "\ntotal kernel time (ticks) = " << total_kernel_ticks << "\n";

    if (clearTick) {
        out << "Stats cleared at tick " << clearTick << "\n";
    }

    // Print GPU PTX file line stats
    // TODO: If running multiple benchmarks in the same simulation, this will
    // need to be updated to print as appropriate
    // printPTXFileLineStats();
}


#if 0
void OpuTop::printPTXFileLineStats() {
    char *temp_ptx_line_stats_filename = ptx_line_stats_filename;
    std::string outfile = simout.directory() + ptx_line_stats_filename;
    ptx_line_stats_filename = (char*)outfile.c_str();
    gpgpu_ctx->stats->ptx_file_line_stats_write_file();
    ptx_line_stats_filename = temp_ptx_line_stats_filename;
}
#endif

void OpuTop::memcpy(void *src, void *dst, size_t count, struct Stream_st *_stream, stream_operation_type type) {
    beginStreamOperation(_stream);
    copyEngine->memcpy((Addr)src, (Addr)dst, count, type);
}

void OpuTop::memcpy_to_symbol(const char *hostVar, const void *src, size_t count, size_t offset, struct Stream_st *_stream) {
    // First, initialize the stream operation
    beginStreamOperation(_stream);

#if 0
    unsigned dst = gpgpu_ctx->func_sim->gpgpu_ptx_hostvar_to_sym_address(hostVar, theOpuUsim);
    // Lookup destination address for transfer:
    std::string sym_name = gpgpu_ctx->gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
    std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
    assert(st != g_sym_name_to_symbol_table.end());
    symbol_table *symtab = st->second;

    symbol *sym = symtab->lookup(sym_name.c_str());
    assert(sym);
    unsigned dst = sym->get_address() + offset;
    printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes to symbol %s+%zu @0x%x ...\n",
           count, sym_name.c_str(), offset, dst);

    copyEngine->memcpy((Addr)src, (Addr)dst, count, stream_memcpy_host_to_device);
#endif
}

void OpuTop::memcpy_from_symbol(void *dst, const char *hostVar, size_t count, size_t offset, struct Stream_st *_stream) {
    // First, initialize the stream operation
    beginStreamOperation(_stream);
#if 0
    unsigned src = gpgpu_ctx->func_sim->gpgpu_ptx_hostvar_to_sym_address(hostVar, theOpuUsim);
    // Lookup destination address for transfer:
    std::string sym_name = gpgpu_ptx_sim_hostvar_to_sym_name(hostVar);
    std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
    assert(st != g_sym_name_to_symbol_table.end());
    symbol_table *symtab = st->second;

    symbol *sym = symtab->lookup(sym_name.c_str());
    assert(sym);
    unsigned src = sym->get_address() + offset;
    printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes from symbol %s+%zu @0x%x ...\n",
           count, sym_name.c_str(), offset, src);


    copyEngine->memcpy((Addr)src, (Addr)dst, count, stream_memcpy_device_to_host);
#endif
}

void OpuTop::memset(Addr dst, int value, size_t count, struct Stream_st *_stream) {
    beginStreamOperation(_stream);
    copyEngine->memset(dst, value, count);
}

void OpuTop::finishCopyOperation()
{
    runningStream->record_next_done();
    scheduleStreamEvent();
    unblockThread(runningTC);
    endStreamOperation();
}

// TODO: When we move the stream manager into libcuda, this will need to be
// eliminated, and libcuda will have to decide when to block the calling thread
bool OpuTop::needsToBlock()
{
    if (!streamManager->empty()) {
        DPRINTF(OpuTop, "Suspend request: Need to activate CPU later\n");
        unblockNeeded = true;
        streamManager->print(stdout);
        return true;
    } else {
        DPRINTF(OpuTop, "Suspend request: Already done.\n");
        return false;
    }
}

void OpuTop::blockThread(ThreadContext *tc, Addr signal_ptr)
{
    if (streamManager->empty()) {
        // It is common in small memcpys for the stream operation to be complete
        // by the time cudaMemcpy calls blockThread. In this case, just signal
        DPRINTF(OpuTop, "No stream operations to block thread %p. Continuing...\n", tc);
        signalThread(tc, signal_ptr);
        blockedThreads.erase(tc);
        unblockNeeded = false;
    } else {
        if (!shaderMMU->isFaultInFlight(tc)) {
            DPRINTF(OpuTop, "Blocking thread %p for GPU syscall\n", tc);
            blockedThreads[tc] = signal_ptr;
            tc->suspend();
        } else {
            DPRINTF(OpuTop, "Pending GPU fault must be handled: Not blocking thread\n");
        }
    }
}

void OpuTop::signalThread(ThreadContext *tc, Addr signal_ptr)
{
    GPUSyscallHelper helper(tc);
    bool signal_val = true;

    // Read signal value and ensure that it is currently false
    // (i.e. thread should be currently blocked)
    helper.readBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
    if (signal_val) {
        panic("Thread doesn't appear to be blocked!\n");
    }

    signal_val = true;
    helper.writeBlob(signal_ptr, (uint8_t*)&signal_val, sizeof(bool));
}

void OpuTop::unblockThread(ThreadContext *tc)
{
    if (!tc) tc = runningTC;
    if (tc->status() != ThreadContext::Suspended) return;
    assert(unblockNeeded);

    if (!streamManager->empty()) {
        // There must be more in the queue of work to complete. Need to
        // continue blocking
        DPRINTF(OpuTop, "Still something in the queue, continuing block\n");
        return;
    }

    DPRINTF(OpuTop, "Unblocking thread %p for GPU syscall\n", tc);
    std::map<ThreadContext*, Addr>::iterator tc_iter = blockedThreads.find(tc);
    if (tc_iter == blockedThreads.end()) {
        panic("Cannot find blocked thread!\n");
    }

    Addr signal_ptr = blockedThreads[tc];
    signalThread(tc, signal_ptr);

    blockedThreads.erase(tc);
    unblockNeeded = false;
    tc->activate();
}

/*
void OpuTop::add_binary( symbol_table *symtab, unsigned fat_cubin_handle )
{
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
}

void OpuTop::add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_info info )
{
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    assert( s != NULL );
    function_info *f = s->get_pc();
    assert( f != NULL );
    f->set_kernel_info(info);
}

void OpuTop::register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun )
{
    if ( m_code.find(fat_cubin_handle) != m_code.end() ) {
        symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
        assert( s != NULL );
        function_info *f = s->get_pc();
        assert( f != NULL );
        // TODO schi hack reset gpgpu_ctx to cuda_gpu
        f->gpgpu_ctx = gpgpu_ctx;
        symbol_table *f_symtab = f->get_symtab();
        f_symtab->update_gpgpu_ctx(gpgpu_ctx);
        f->ptx_assemble();
        m_kernel_lookup[hostFun] = f;
    } else {
        m_kernel_lookup[hostFun] = NULL;
    }
}

function_info *OpuTop::get_kernel(const char *hostFun)
{
    std::map<const void*,function_info*>::iterator i = m_kernel_lookup.find(hostFun);
    if (i != m_kernel_lookup.end()) {
        return i->second;
    }
    return NULL;
}
*/

void OpuTop::setInstBaseVaddr(uint64_t addr)
{
    if (!instBaseVaddrSet) {
        instBaseVaddr = addr;
        instBaseVaddrSet = true;
    }
}

uint64_t OpuTop::getInstBaseVaddr()
{
    return instBaseVaddr;
}

void OpuTop::setLocalBaseVaddr(Addr addr)
{
    assert(!localBaseVaddr);
    localBaseVaddr = addr;
}

uint64_t OpuTop::getLocalBaseVaddr()
{
    if (!localBaseVaddr) {
        panic("Local base virtual address is unset!"
              " Make sure bench was compiled with latest libcuda.\n");
    }
    return localBaseVaddr;
}

Addr OpuTop::GPUPageTable::addrToPage(Addr addr)
{
    Addr offset = addr % TheISA::PageBytes;
    return addr - offset;
}

void OpuTop::GPUPageTable::serialize(CheckpointOut &cp) const
{
    unsigned int num_ptes = pageMap.size();
    unsigned int index = 0;
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    std::map<Addr, Addr>::const_iterator it = pageMap.begin();
    for (; it != pageMap.end(); ++it) {
        pagetable_vaddrs[index] = (*it).first;
        pagetable_paddrs[index++] = (*it).second;
    }
    SERIALIZE_SCALAR(num_ptes);
    SERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    SERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void OpuTop::GPUPageTable::unserialize(CheckpointIn &cp)
{
    unsigned int num_ptes = 0;
    UNSERIALIZE_SCALAR(num_ptes);
    Addr* pagetable_vaddrs = new Addr[num_ptes];
    Addr* pagetable_paddrs = new Addr[num_ptes];
    UNSERIALIZE_ARRAY(pagetable_vaddrs, num_ptes);
    UNSERIALIZE_ARRAY(pagetable_paddrs, num_ptes);
    for (unsigned int i = 0; i < num_ptes; ++i) {
        pageMap[pagetable_vaddrs[i]] = pagetable_paddrs[i];
    }
    delete[] pagetable_vaddrs;
    delete[] pagetable_paddrs;
}

void OpuTop::registerDeviceMemory(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory || accessHostPageTable) return;
    DPRINTF(OpuTopPageTable, "Registering device memory vaddr: %x, size: %d\n", vaddr, size);
    // Get the physical address of full memory allocation (i.e. all pages)
    Addr page_vaddr, page_paddr;
    bool success = true;
    for (ChunkGenerator gen(vaddr, size, TheISA::PageBytes); !gen.done(); gen.next()) {
        page_vaddr = pageTable.addrToPage(gen.addr());
        if (FullSystem) {
            // panic("FIXME: need to find a way for vtophys call");
            // page_paddr = TheISA::vtophys(tc, page_vaddr);
            tc->getProcessPtr()->pTable->translate(page_vaddr, page_paddr);
        } else {
            success = tc->getProcessPtr()->pTable->translate(page_vaddr, page_paddr);
            if (!success) {
                printf("OpuTop registerDeviceMemory translate vaddr %lx failed\n", page_vaddr);
                panic("registerDeviceMemory error");
            }
        }
        pageTable.insert(page_vaddr, page_paddr);
    }
}

void OpuTop::registerDeviceInstText(ThreadContext *tc, Addr vaddr, size_t size)
{
    if (manageGPUMemory) {
        // Allocate virtual and physical memory for the device text
        Addr gpu_vaddr = allocateGPUMemory(size);
        setInstBaseVaddr(gpu_vaddr);
    } else {
        setInstBaseVaddr(vaddr);
        registerDeviceMemory(tc, vaddr, size);
    }
}

Addr OpuTop::allocateGPUMemory(size_t size)
{
    assert(manageGPUMemory);
    DPRINTF(OpuTopPageTable, "GPU allocating %d bytes\n", size);

    if (size == 0) return 0;

    // TODO: When we need to reclaim memory, this will need to be modified
    // heavily to actually track allocated and free physical and virtual memory
    bool alloc_cp_memory = false;

    if (virtualGPUBrkAddr == cpMemoryBaseVaddr) {
        alloc_cp_memory = true;
        size += cpMemoryBaseSize;
    }

    // Cache block align the allocation size
    size_t block_part = size % ruby->getBlockSizeBytes();
    size_t aligned_size = size + (block_part ? (ruby->getBlockSizeBytes() - block_part) : 0);

    aligned_size = roundUp(aligned_size, TheISA::PageBytes);

    Addr base_vaddr = virtualGPUBrkAddr;
    virtualGPUBrkAddr += aligned_size;
    // Addr base_paddr = physicalGPUBrkAddr;
    // physicalGPUBrkAddr += aligned_size;

    if (virtualGPUBrkAddr > gpuMemoryRange.size()) {
        panic("Ran out of GPU memory!");
    }

    int aligned_pages = divCeil(aligned_size, TheISA::PageBytes);
    Addr base_paddr = system->allocPhysPages(aligned_pages);

    Addr page_vaddr = pageTable.addrToPage(base_vaddr);
    if (page_vaddr < base_vaddr) {
        base_paddr = base_paddr + (base_vaddr - page_vaddr);
    }

    // Map pages to physical pages
    for (ChunkGenerator gen(base_vaddr, aligned_size, TheISA::PageBytes); !gen.done(); gen.next()) {
        Addr page_vaddr = pageTable.addrToPage(gen.addr());
        Addr page_paddr;
        if (page_vaddr <= base_vaddr) {
            page_paddr = base_paddr - (base_vaddr - page_vaddr);
        } else {
            page_paddr = base_paddr + (page_vaddr - base_vaddr);
        }
        DPRINTF(OpuTopPageTable, "  Trying to allocate page at vaddr %x with addr %x\n", page_vaddr, gen.addr());
        pageTable.insert(page_vaddr, page_paddr);
    }

    DPRINTF(OpuTopAccess, "Allocating %d bytes for GPU at address 0x%x\n", size, base_vaddr);

    if (alloc_cp_memory) {
        base_vaddr += cpMemoryBaseSize;
        // FIXME commandProcessor->active = true;
    }
    return base_vaddr;
}

void OpuTop::regStats()
{

    ClockedObject::regStats();
    using namespace Stats;

    numKernelsStarted
        .name(name() + ".kernels_started")
        .desc("Number of kernels started");
    numKernelsCompleted
        .name(name() + ".kernels_completed")
        .desc("Number of kernels completed");

    // it is called in addStatGroup
    // copyEngine->regStats();
    for (auto iter = opuCores.begin(); iter != opuCores.end(); ++iter) {
        // (*iter)->regStats();
    }
    // shaderMMU->regStats();
}
/*
OpuSimComponentWrapper *OpuSimComponentWrapperParams::create() const {
    return new OpuSimComponentWrapper(*this);
}
*/

/**
* virtual process function that is invoked when the callback
* queue is executed.
*/
/*
void GPUExitCallback::process()
{
    // TODO std::ostream *os = simout.find(stats_filename);
    OutputStream *os = simout.find(stats_filename);
    if (!os) {
        os = simout.create(stats_filename);
    }
    gpu->gpuPrintStats(*os->stream());
    *os->stream() << std::endl;
}
*/
void OpuTop::callback()
{
    this->gpuPrintStats(*statsFile->stream());
    *statsFile->stream() << std::endl;
}

}
