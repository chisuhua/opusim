#pragma once
#include "gpu-cache.h"

// TODO schi #include "base/misc.hh"
#include "gem5/opu_core.hh"
#include "gem5/opu_top.hh"

namespace {
class OpuCore;
}

class const_cache : public read_only_cache {
    abstract_core* abstractGPU;
    gem5::OpuCore* shaderCore;
    unsigned m_sid;
public:
    const_cache(abstract_core* _gpu, const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status);
    enum cache_request_status access(new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    void cycle();
};

