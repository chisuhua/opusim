#pragma once
#include "cache_base.h"

// TODO schi #include "base/misc.hh"
// #include "gem5/opu_core.hh"
// #include "gem5/opu_top.hh"

class core_t;
namespace gem5 {
class OpuCore;
}

class l0_ccache : public read_only_cache {
    core_t* m_core;
    gem5::OpuCore* shaderCore;
    unsigned m_sid;
public:
    l0_ccache(core_t* _gpu, const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status);
    enum cache_request_status access(address_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    void cycle();
};

