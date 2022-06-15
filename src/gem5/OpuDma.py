# Copyright (c) 2011 Mark D. Hill and David A. Wood
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *
#from m5.objects.MemObject import MemObject
from m5.objects.ClockedObject import ClockedObject
from m5.objects.OpuTlb import OpuTLB

class OpuDma(ClockedObject):
    type = 'OpuDma'
    cxx_class = 'gem5::OpuDma'
    cxx_header = "opu_dma.hh"

    host_port = MasterPort("The copy engine port to host coherence domain")
    device_port = MasterPort("The copy engine port to device coherence domain")
    driver_delay = Param.Int(0, "memcpy launch delay in ticks");
    sys = Param.System(Parent.any, "system sc will run on")
    # @TODO: This will need to be removed when CUDA syscalls manage copies
    opu = Param.OpuTop(Parent.any, "The OPU")

    cache_line_size = Param.Unsigned(Parent.cache_line_size, "Cache line size in bytes")
    buffering = Param.Unsigned(0, "The maximum cache lines that the copy engine"
                                  "can buffer (0 implies effectively infinite)")

    host_dtb = Param.OpuTLB(OpuTLB(access_host_pagetable = True), "TLB for the host memory space")
    device_dtb = Param.OpuTLB(OpuTLB(), "TLB for the device memory space")

    id = Param.Int(-1, "ID of the CE")
    stats_filename = Param.String("ce_stats.txt",
        "file to which copy engine dumps its stats")
