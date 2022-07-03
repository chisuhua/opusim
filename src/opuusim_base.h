#pragma once
#include <stdint.h>
#include <memory>

class opu_sim_config;
class KernelInfo;
namespace gem5 {
class OpuCoreBase;
class OpuTop;
class KernelInfoBase;

class OpuSimBase {
public:
    OpuSimBase(OpuTop *opu_top): opu_top(opu_top) {};
    virtual OpuCoreBase* getSIMTCore(uint32_t id) = 0;
    virtual void init() = 0;
    virtual bool active() = 0;
    // virtual void startCycleFunction() = 0;
    // virtual void endCycleFunction() = 0;
    virtual uint32_t shared_mem_size() const = 0;
    virtual uint32_t num_registers_per_core() const = 0;
    virtual int warp_size() const = 0;
    virtual uint32_t shader_clock() const = 0;
    virtual void core_cycle_start() = 0;
    virtual void core_cycle_end() = 0;
    virtual const opu_sim_config *get_config() const = 0;
    virtual bool can_start_kernel() = 0;
    virtual void launch(gem5::KernelInfoBase *) = 0;
    virtual uint32_t finished_kernel() = 0;
    virtual void stop_all_running_kernels() = 0;
    virtual void print_stats() = 0;
    virtual bool cycle_insn_cta_max_hit() = 0;
    // virtual void set_cache_config() = 0;
    //
    virtual uint32_t num_cores() = 0;

    unsigned long long gpu_sim_cycle;
    unsigned long long gpu_tot_sim_cycle;
    OpuTop* get_opu() {return opu_top;}

    OpuTop *opu_top;
};

class KernelInfoBase {
public:
    // KernelInfoBase(std::string& name) : name_(name) {};
    KernelInfoBase() {};
    virtual ~KernelInfoBase() {};
    uint32_t m_launch_latency {0};
    // std::string name() { return name_;};
    uint32_t get_uid() { return uid;};
    void print_parent_info() {};
    bool is_finished() { return finished;};
    void notify_parent_finished() {};
    // std::string name_;
    uint32_t uid;
    bool finished;
};

}
