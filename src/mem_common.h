const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
const unsigned SECTOR_CHUNCK_SIZE = 4;   //four sectors
const unsigned SECTOR_SIZE = 32 ;        //sector is 32 bytes width
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;

class mem_access_t {
 public:
  mem_access_t(gpgpu_context *ctx) { init(ctx); }
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, gpgpu_context *ctx) {
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),
        m_byte_mask(byte_mask),
        m_sector_mask(sector_mask) {
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }

  new_addr_type get_addr() const { return m_addr; }
  void set_addr(new_addr_type addr) { m_addr = addr; }
  unsigned get_size() const { return m_req_size; }
  const active_mask_t &get_warp_mask() const { return m_warp_mask; }
  bool is_write() const { return m_write; }
  enum mem_access_type get_type() const { return m_type; }
  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

  void print(FILE *fp) const {
    fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
            m_write ? "store" : "load ", m_req_size);
    switch (m_type) {
      case GLOBAL_ACC_R:
        fprintf(fp, "GLOBAL_R");
        break;
      case LOCAL_ACC_R:
        fprintf(fp, "LOCAL_R ");
        break;
      case CONST_ACC_R:
        fprintf(fp, "CONST   ");
        break;
      case TEXTURE_ACC_R:
        fprintf(fp, "TEXTURE ");
        break;
      case GLOBAL_ACC_W:
        fprintf(fp, "GLOBAL_W");
        break;
      case LOCAL_ACC_W:
        fprintf(fp, "LOCAL_W ");
        break;
      case L2_WRBK_ACC:
        fprintf(fp, "L2_WRBK ");
        break;
      case INST_ACC_R:
        fprintf(fp, "INST    ");
        break;
      case L1_WRBK_ACC:
        fprintf(fp, "L1_WRBK ");
        break;
      default:
        fprintf(fp, "unknown ");
        break;
    }
  }

  gpgpu_context *gpgpu_ctx;

private:
  void init(gpgpu_context *ctx);
  /*
   void init() 
   {
      m_uid=++sm_next_access_uid;
      m_addr=0;
      m_req_size=0;
   }
   */

  unsigned m_uid;
  new_addr_type m_addr;  // request address
  bool m_write;
  unsigned m_req_size;  // bytes
  mem_access_type m_type;
  active_mask_t m_warp_mask;
  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;

  static unsigned sm_next_access_uid;
};

class mem_fetch;

class mem_fetch_interface {
 public:
  virtual bool full(unsigned size, bool write) const = 0;
  virtual void push(mem_fetch *mf) = 0;
};

class mem_fetch_allocator {
public:
  /*
  virtual mem_fetch *alloc( new_addr_type addr, mem_access_type type,
                                unsigned size, bool wr ) const = 0;
  virtual mem_fetch *alloc( const class warp_inst_t &inst,
                                const mem_access_t &access ) const = 0;
                                */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr,
                           unsigned long long cycle) const = 0;
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const = 0;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf) const = 0;
};

