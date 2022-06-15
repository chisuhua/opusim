#include "opusim_base.hh"

std::string KernelInfoBase::name() {
    return name;
}

uint32_t KernelInfoBase::get_uid() {
    return uid;
}

void KernelInfoBase::print_parent_info() {
}

bool KernelInfoBase::is_finished() {
    return finished;
}

void KernelInfoBase::notify_parent_finished() {
};
