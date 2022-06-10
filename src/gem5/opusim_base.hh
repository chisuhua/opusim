#ifndef __OPUSIM_BASE_HH__
#define __OPUSIM_BASE_HH__

class OpuCoreBase;

class OpuSimBase {
public:
    virtual ~OpuSimBase() {};
    virtual OpuCoreBase* getCore(uint32_t id) = 0;
    virtual void init() = 0;
    virtual bool active() = 0;
    virtual void startCycleFunction() = 0;
    virtual void endCycleFunction() = 0;
    virtual uint32_t shared_mem_size() = 0;
    virtual uint32_t num_registers_per_core() = 0;
    virtual uint32_t wrp_size() = 0;
    virtual uint32_t shader_clock() = 0;
    virtual void core_cycle_start() = 0;
    virtual void core_cycle_end() = 0;
    virtual uint32_t get_config() = 0;
};

#endif
