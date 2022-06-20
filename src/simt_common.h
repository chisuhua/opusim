#pragma once
#include <stdint.h>
#include <vector>
#include <map>
#include <bitset>
//#include "core/warp_inst.h"

using address_type = uint64_t;

const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK  MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

const unsigned WARP_PER_CTA_MAX = 64;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;


//enum divergence_support_t {
//   POST_DOMINATOR = 1,
//   NUM_SIMD_MODEL
//};

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;
const unsigned MAX_DATA_BYTES_PER_INSN_PER_THREAD = 16;

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

//After expanding the vector input and output operands 
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;
// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;
// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;
// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;
// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;
const unsigned long long TOTAL_LOCAL_MEM_PER_SM = MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long TOTAL_SHARED_MEM = MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
const unsigned long long TOTAL_LOCAL_MEM = MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;

enum class barrier_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
enum class reduction_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };

// enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
// typedef enum uarch_operand_type_t types_of_operands;

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32

enum pipeline_stage_name_t {
  ID_OC_SP = 0,
  ID_OC_DP,
  ID_OC_INT,
  ID_OC_SFU,
  ID_OC_MEM,
  OC_EX_SP,
  OC_EX_DP,
  OC_EX_INT,
  OC_EX_SFU,
  OC_EX_MEM,
  EX_WB,
  ID_OC_TENSOR_CORE,
  OC_EX_TENSOR_CORE,
  N_PIPELINE_STAGES
};

const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};

enum FuncCache {
  FuncCachePreferNone = 0,
  FuncCachePreferShared = 1,
  FuncCachePreferL1 = 2
};

enum AdaptiveCache { FIXED = 0, ADAPTIVE_CACHE = 1 };


// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

enum TRACE {
    WARP_SCHEDULER = 0,
    SCOREBOARD ,
    MEMORY_PARTITION_UNIT ,
    MEMORY_SUBPARTITION_UNIT ,
    INTERCONNECT ,
    LIVENESS ,
    NUM_TRACE_STREAMS
};

#define SHADER_DTRACE(x)  (false)
#define SHADER_DPRINTF(x, ...) do {} while (0)
#define SCHED_DPRINTF(x, ...) do {} while (0)


