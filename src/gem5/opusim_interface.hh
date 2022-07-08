
namespace opu {
class opu_sim_config;
}

struct DispatchInfo;

namespace gem5 {
class OpuSimBase;
class KernelInfoBase;
class OpuContext;
class OpuTop;
}

namespace gem5 {

extern "C" typedef OpuSimBase* (*pfn_make_opusim)(opu::opu_sim_config* &config, gem5::OpuContext *ctx, gem5::OpuTop *);
extern "C" typedef KernelInfoBase* (*pfn_make_kernel)(DispatchInfo *disp_info);

class OpuSimInterface {
public:
    pfn_make_opusim make_opusim;
    pfn_make_kernel make_kernel;
    OpuSimInterface();

    static OpuSimInterface* GetInstance() {
        static OpuSimInterface *interface = nullptr;
        if (interface == nullptr) {
            interface = new OpuSimInterface();
        }
        return interface;
    }
};

}

