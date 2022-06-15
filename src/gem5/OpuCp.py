# Copyright (c) 2015 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2005-2007 The Regents of The University of Michigan
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
# Authors: Nathan Binkert
#          Andreas Hansson
from m5.SimObject import *
from m5.defines import buildEnv
from m5.params import *
from m5.proxy import *
from m5.util.fdthelper import *

#from OpuTLB import OpuTLB
from m5.objects.ClockedObject import ClockedObject
from m5.objects.OpuTlb import OpuTLB

class OpuCp(ClockedObject):
    type = 'OpuCp'
    cxx_class = 'gem5::OpuCp'
    cxx_header = "opu_cp.hh"

    host_port = MasterPort("The CP port to host coherence domain")
    device_port = MasterPort("The CP engine port to device coherence domain")
    driver_delay = Param.Int(0, "memcpy launch delay in ticks");

    pio_port = SlavePort("The CP I/O port")
    pio_addr = Param.Addr(0x300000000, "Device Address")
    pio_latency = Param.Latency('1ns', "Programmed IO latency")

    system = Param.System(Parent.any, "System this cp is part of")

    # @TODO: This will need to be removed when CUDA syscalls manage copies
    opu = Param.OpuTop(Parent.any, "The GPU")
    file_name = Param.String("zephyr so file")

    cache_line_size = Param.Unsigned(Parent.cache_line_size, "Cache line size in bytes")
    buffering = Param.Unsigned(0, "The maximum cache lines that the copy engine"
                                  "can buffer (0 implies effectively infinite)")

    host_dtb = Param.OpuTLB(OpuTLB(access_host_pagetable = True), "TLB for the host memory space")
    device_dtb = Param.OpuTLB(OpuTLB(), "TLB for the device memory space")

    max_loads = Param.Counter(0, "Number of loads to execute before exiting")

    start_tick = Param.Counter(0x10000, "the tick to start test)")
    os_event_tick = Param.Counter(0x10000, "the OS event tick)")

    # Add the ability to supress error responses on functional
    # accesses as Ruby needs this
    suppress_func_warnings = Param.Bool(False, "Suppress warnings when "\
                                            "functional accesses fail.")
