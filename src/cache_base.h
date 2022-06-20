#pragma once
#include <list>

struct cache_event {
  enum cache_event_type m_cache_event_type;
  evicted_block_info m_evicted_block;  // if it was write_back event, fill the
                                       // the evicted block info

  cache_event(enum cache_event_type m_cache_event) {
    m_cache_event_type = m_cache_event;
  }

  cache_event(enum cache_event_type cache_event,
              evicted_block_info evicted_block) {
    m_cache_event_type = cache_event;
    m_evicted_block = evicted_block;
  }
};

class cache_t {
 public:
  virtual ~cache_t() {}
  virtual enum cache_request_status access(address_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;

  // accessors for cache bandwidth availability
  virtual bool data_port_free() const = 0;
  virtual bool fill_port_free() const = 0;
};

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
class baseline_cache : public cache_t {
 public:
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status)
      : m_config(config),
        m_tag_array(new tag_array(config, core_id, type_id)),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config) {
    init(name, config, memport, status);
  }

  void init(const char *name, const cache_config &config,
            mem_fetch_interface *memport, enum mem_fetch_status status) {
    m_name = name;
    assert(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
    m_memport = memport;
    m_miss_queue_status = status;
  }

  virtual ~baseline_cache() { delete m_tag_array; }

  void update_cache_parameters(cache_config &config) {
    m_config = config;
    m_tag_array->update_cache_parameters(config);
    m_mshrs.check_mshr_parameters(config.m_mshr_entries,
                                  config.m_mshr_max_merge);
  }

  virtual enum cache_request_status access(address_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;
  /// Sends next request to lower level of memory
  void cycle();
  /// Interface for response from lower memory level (model bandwidth
  /// restictions in caller)
  void fill(mem_fetch *mf, unsigned time);
  /// Checks if mf is waiting to be filled by lower memory level
  bool waiting_for_fill(mem_fetch *mf);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  bool access_ready() const { return m_mshrs.access_ready(); }
  /// Pop next ready access (does not include accesses that "HIT")
  mem_fetch *next_access() { return m_mshrs.next_access(); }
  // flash invalidate all entries in cache
  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }
  void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
  void display_state(FILE *fp) const;

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }
  void get_sub_stats(struct cache_sub_stats &css) const {
    m_stats.get_sub_stats(css);
  }
  // Clear per-window stats for AerialVision support
  void clear_pw() { m_stats.clear_pw(); }
  // Per-window sub stats for AerialVision support
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
    m_stats.get_sub_stats_pw(css);
  }

  // accessors for cache bandwidth availability
  bool data_port_free() const {
    return m_bandwidth_management.data_port_free();
  }
  bool fill_port_free() const {
    return m_bandwidth_management.fill_port_free();
  }

  // This is a gapping hole we are poking in the system to quickly handle
  // filling the cache on cudamemcopies. We don't care about anything other than
  // L2 state after the memcopy - so just force the tag array to act as though
  // something is read or written without doing anything else.
  void force_tag_access(address_type addr, unsigned time,
                        mem_access_sector_mask_t mask) {
    mem_access_byte_mask_t byte_mask;
    m_tag_array->fill(addr, time, mask, byte_mask, true);
  }

 protected:
  // Constructor that can be used by derived classes with custom tag arrays
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, tag_array *new_tag_array)
      : m_config(config),
        m_tag_array(new_tag_array),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config) {
    init(name, config, memport, status);
  }

 protected:
  std::string m_name;
  cache_config &m_config;
  tag_array *m_tag_array;
  mshr_table m_mshrs;
  std::list<mem_fetch *> m_miss_queue;
  enum mem_fetch_status m_miss_queue_status;
  mem_fetch_interface *m_memport;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(address_type a, address_type ad, unsigned i, unsigned d,
                    const cache_config &m_config) {
      m_valid = true;
      m_block_addr = a;
      m_addr = ad;
      m_cache_index = i;
      m_data_size = d;
      pending_read = m_config.m_mshr_type == SECTOR_ASSOC
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    address_type m_block_addr;
    address_type m_addr;
    unsigned m_cache_index;
    unsigned m_data_size;
    // this variable is used when a load request generates multiple load
    // transactions For example, a read request from non-sector L1 request sends
    // a request to sector L2
    unsigned pending_read;
  };

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;

  cache_stats m_stats;

  /// Checks whether this request can be handled on this cycle. num_miss equals
  /// max # of misses to be handled on this cycle
  bool miss_queue_full(unsigned num_miss) {
    return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  }
  /// Read miss handler without writeback
  void send_read_request(address_type addr, address_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, std::list<cache_event> &events,
                         bool read_only, bool wa);
  /// Read miss handler. Check MSHR hit or MSHR available
  void send_read_request(address_type addr, address_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, bool &wb, evicted_block_info &evicted,
                         std::list<cache_event> &events, bool read_only,
                         bool wa);

  /// Sub-class containing all metadata for port bandwidth management
  class bandwidth_management {
   public:
    bandwidth_management(cache_config &config);

    /// use the data port based on the outcome and events generated by the
    /// mem_fetch request
    void use_data_port(mem_fetch *mf, enum cache_request_status outcome,
                       const std::list<cache_event> &events);

    /// use the fill port
    void use_fill_port(mem_fetch *mf);

    /// called every cache cycle to free up the ports
    void replenish_port_bandwidth();

    /// query for data port availability
    bool data_port_free() const;
    /// query for fill port availability
    bool fill_port_free() const;

   protected:
    const cache_config &m_config;

    int m_data_port_occupied_cycles;  //< Number of cycle that the data port
                                      // remains used
    int m_fill_port_occupied_cycles;  //< Number of cycle that the fill port
                                      // remains used
  };

  bandwidth_management m_bandwidth_management;
};

/// Read only cache
class read_only_cache : public baseline_cache {
 public:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status)
      : baseline_cache(name, config, core_id, type_id, memport, status) {}

  /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
  /// could not be accepted (for any reason)
  virtual enum cache_request_status access(address_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

  virtual ~read_only_cache() {}

 protected:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, tag_array *new_tag_array)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {}
};

/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
 public:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, mem_access_type wr_alloc_type,
             mem_access_type wrbk_type, class gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  virtual ~data_cache() {}

  virtual void init(mem_fetch_allocator *mfcreator) {
    m_memfetch_creator = mfcreator;

    // Set read hit function
    m_rd_hit = &data_cache::rd_hit_base;

    // Set read miss function
    m_rd_miss = &data_cache::rd_miss_base;

    // Set write hit function
    switch (m_config.m_write_policy) {
      // READ_ONLY is now a separate cache class, config is deprecated
      case READ_ONLY:
        assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
        break;
      case WRITE_BACK:
        m_wr_hit = &data_cache::wr_hit_wb;
        break;
      case WRITE_THROUGH:
        m_wr_hit = &data_cache::wr_hit_wt;
        break;
      case WRITE_EVICT:
        m_wr_hit = &data_cache::wr_hit_we;
        break;
      case LOCAL_WB_GLOBAL_WT:
        m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
        break;
      default:
        assert(0 && "Error: Must set valid cache write policy\n");
        break;  // Need to set a write hit function
    }

    // Set write miss function
    switch (m_config.m_write_alloc_policy) {
      case NO_WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_no_wa;
        break;
      case WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_wa_naive;
        break;
      case FETCH_ON_WRITE:
        m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
        break;
      case LAZY_FETCH_ON_READ:
        m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
        break;
      default:
        assert(0 && "Error: Must set valid cache write miss policy\n");
        break;  // Need to set a write miss function
    }
  }

  virtual enum cache_request_status access(address_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

 protected:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, tag_array *new_tag_array,
             mem_access_type wr_alloc_type, mem_access_type wrbk_type,
             class gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  mem_access_type m_wr_alloc_type;  // Specifies type of write allocate request
                                    // (e.g., L1 or L2)
  mem_access_type
      m_wrbk_type;  // Specifies type of writeback request (e.g., L1 or L2)
  class gpgpu_sim *m_gpu;

  //! A general function that takes the result of a tag_array probe
  //  and performs the correspding functions based on the cache configuration
  //  The access fucntion calls this function
  enum cache_request_status process_tag_probe(bool wr,
                                              enum cache_request_status status,
                                              address_type addr,
                                              unsigned cache_index,
                                              mem_fetch *mf, unsigned time,
                                              std::list<cache_event> &events);

 protected:
  mem_fetch_allocator *m_memfetch_creator;

  // Functions for data cache access
  /// Sends write request to lower level memory (write or writeback)
  void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
                          std::list<cache_event> &events);
  void update_m_readable(mem_fetch *mf, unsigned cache_index);
  // Member Function pointers - Set by configuration options
  // to the functions below each grouping
  /******* Write-hit configs *******/
  enum cache_request_status (data_cache::*m_wr_hit)(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Marks block as MODIFIED and updates block LRU
  enum cache_request_status wr_hit_wb(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-back
  enum cache_request_status wr_hit_wt(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-through

  /// Marks block as INVALID and sends write request to lower level memory
  enum cache_request_status wr_hit_we(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-evict
  enum cache_request_status wr_hit_global_we_local_wb(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  // global write-evict, local write-back

  /******* Write-miss configs *******/
  enum cache_request_status (data_cache::*m_wr_miss)(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Sends read request, and possible write-back request,
  //  to lower level memory for a write miss with write-allocate
  enum cache_request_status wr_miss_wa_naive(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate-send-write-and-read-request
  enum cache_request_status wr_miss_wa_fetch_on_write(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate with fetch-on-every-write
  enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-allocate with read-fetch-only
  enum cache_request_status wr_miss_wa_write_validate(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate that writes with no read fetch
  enum cache_request_status wr_miss_no_wa(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // no write-allocate

  // Currently no separate functions for reads
  /******* Read-hit configs *******/
  enum cache_request_status (data_cache::*m_rd_hit)(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_hit_base(address_type addr,
                                        unsigned cache_index, mem_fetch *mf,
                                        unsigned time,
                                        std::list<cache_event> &events,
                                        enum cache_request_status status);

  /******* Read-miss configs *******/
  enum cache_request_status (data_cache::*m_rd_miss)(
      address_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_miss_base(address_type addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned time,
                                         std::list<cache_event> &events,
                                         enum cache_request_status status);
};

