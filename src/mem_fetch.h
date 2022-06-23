#pragma once
#include "mem_common.h"
#include "core/warp_inst.h"
#include "opucore_base.h"

enum mf_type {
  READ_REQUEST = 0,
  WRITE_REQUEST,
  READ_REPLY,  // send to shader
  WRITE_ACK
};

enum mem_fetch_status {
    MEM_FETCH_INITIALIZED = 0,
    IN_L1I_MISS_QUEUE,
    IN_L1D_MISS_QUEUE,
    IN_L1T_MISS_QUEUE,
    IN_L1C_MISS_QUEUE,
    IN_L1TLB_MISS_QUEUE,
    IN_VM_MANAGER_QUEUE,
    IN_ICNT_TO_MEM,
    IN_PARTITION_ROP_DELAY,
    IN_PARTITION_ICNT_TO_L2_QUEUE,
    IN_PARTITION_L2_TO_DRAM_QUEUE,
    IN_PARTITION_DRAM_LATENCY_QUEUE,
    IN_PARTITION_L2_MISS_QUEUE,
    IN_PARTITION_MC_INTERFACE_QUEUE,
    IN_PARTITION_MC_INPUT_QUEUE,
    IN_PARTITION_MC_BANK_ARB_QUEUE,
    IN_PARTITION_DRAM,
    IN_PARTITION_MC_RETURNQ,
    IN_PARTITION_DRAM_TO_L2_QUEUE,
    IN_PARTITION_L2_FILL_QUEUE,
    IN_PARTITION_L2_TO_ICNT_QUEUE,
    IN_ICNT_TO_SHADER,
    IN_CLUSTER_TO_SHADER_QUEUE,
    IN_SHADER_LDST_RESPONSE_FIFO,
    IN_SHADER_FETCHED,
    IN_SHADER_L1T_ROB,
    MEM_FETCH_DELETED,
    NUM_MEM_REQ_STAT
};

class mem_fetch : public gem5::OpuMemfetch {
 public:
  mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
            unsigned ctrl_size, unsigned wid, unsigned sid, unsigned tpc,
            unsigned long long cycle,
            mem_fetch *original_mf = NULL, mem_fetch *original_wr_mf = NULL);
  ~mem_fetch();

  void set_status(enum mem_fetch_status status, unsigned long long cycle);
  void set_reply() {
    assert(m_access.get_type() != L1_WRBK_ACC &&
           m_access.get_type() != L2_WRBK_ACC);
    if (m_type == READ_REQUEST) {
      assert(!get_is_write());
      m_type = READ_REPLY;
    } else if (m_type == WRITE_REQUEST) {
      assert(get_is_write());
      m_type = WRITE_ACK;
    }
  }
  void do_atomic();

  void print(FILE *fp, bool print_inst = true) const;

  // const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
  // void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }
  // void set_parition(unsigned sub_partition_id) {
  //   m_raw_addr.sub_partition = sub_partition_id;
  // }
  unsigned get_data_size() const { return m_data_size; }
  void set_data_size(unsigned size) { m_data_size = size; }
  unsigned get_ctrl_size() const { return m_ctrl_size; }
  unsigned size() const { return m_data_size + m_ctrl_size; }
  bool is_write() { return m_access.is_write(); }
  void set_addr(address_type addr) { m_access.set_addr(addr); }
  address_type get_addr() const { return m_access.get_addr(); }
  unsigned get_access_size() const { return m_access.get_size(); }
  // address_type get_partition_addr() const { return m_partition_addr; }
  // unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
  bool get_is_write() const { return m_access.is_write(); }
  unsigned get_request_uid() const { return m_request_uid; }
  unsigned get_sid() const { return m_sid; }
  unsigned get_tpc() const { return m_tpc; }
  unsigned get_wid() const { return m_wid; }
  bool istexture() const;
  bool isconst() const;
  enum mf_type get_type() const { return m_type; }
  bool isatomic() const;

  void set_return_timestamp(unsigned t) { m_timestamp2 = t; }
  void set_icnt_receive_time(unsigned t) { m_icnt_receive_time = t; }
  unsigned get_timestamp() const { return m_timestamp; }
  unsigned get_return_timestamp() const { return m_timestamp2; }
  unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

  enum mem_access_type get_access_type() const { return m_access.get_type(); }
  const active_mask_t &get_access_warp_mask() const {
    return m_access.get_warp_mask();
  }
  mem_access_byte_mask_t get_access_byte_mask() const {
    return m_access.get_byte_mask();
  }
  mem_access_sector_mask_t get_access_sector_mask() const {
    return m_access.get_sector_mask();
  }

  address_type get_pc() const { return m_inst.empty() ? -1 : m_inst.pc; }
  const warp_inst_t &get_inst() { return m_inst; }
  enum mem_fetch_status get_status() const { return m_status; }

  unsigned get_num_flits(bool simt_to_mem);

  mem_fetch *get_original_mf() { return original_mf; }
  mem_fetch *get_original_wr_mf() { return original_wr_mf; }

private:
  // request source information
  unsigned m_request_uid;
  unsigned m_sid;
  unsigned m_tpc;
  unsigned m_wid;

  // where is this request now?
  enum mem_fetch_status m_status;
  unsigned long long m_status_change;

  // request type, address, size, mask
  mem_access_t m_access;
  unsigned m_data_size;  // how much data is being written
  unsigned
      m_ctrl_size;  // how big would all this meta data be in hardware (does not
                    // necessarily match actual size of mem_fetch)
  // address_type
  //    m_partition_addr;  // linear physical address *within* dram partition
                         // (partition bank select bits squeezed out)
  // addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
  //                       // chip-row-bank-column address)
  enum mf_type m_type;

  // statistics
  unsigned
      m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
  unsigned m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
                          // onto icnt to shader; only used for reads
  unsigned m_icnt_receive_time;  // set to gpu_sim_cycle + interconnect_latency
                                 // when fixed icnt latency mode is enabled

  // requesting instruction (put last so mem_fetch prints nicer in gdb)
  warp_inst_t m_inst;

  static unsigned sm_next_mf_request_uid;

  unsigned icnt_flit_size;

  mem_fetch
      *original_mf;  // this pointer is set up when a request is divided into
                     // sector requests at L2 cache (if the req size > L2 sector
                     // size), so the pointer refers to the original request
  mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
                              // when fetch-on-write policy is used
};

