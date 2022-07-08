#include "opusim_interface.hh"
#include <stdio.h>
#include <dlfcn.h>
#include <cassert>


namespace gem5 {
OpuSimInterface::OpuSimInterface() {
    void* lib_handle = dlopen("libopusim.so", RTLD_NOW | RTLD_GLOBAL);
    if (lib_handle == nullptr) {
        printf("Failed to load libopusim.so, error - %sn\n", dlerror());
        assert(false);
    }

    pfn_make_opusim make_opusim = (pfn_make_opusim)dlsym(lib_handle, "make_opusim");
    if (make_opusim == nullptr) {
        printf("Failed to dlsym make_opusim, error - %sn\n", dlerror());
        assert(false);
    }

    pfn_make_kernel make_kernel = (pfn_make_kernel)dlsym(lib_handle, "make_kernel");
    if (make_kernel == nullptr) {
        printf("Failed to dlsym make_kernel, error - %sn\n", dlerror());
        assert(false);
    }
};
}

