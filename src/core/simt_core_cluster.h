class simt_core_cluster {
 public:
  simt_core_cluster(class opu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    shader_core_stats *stats);

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);
  unsigned get_not_completed() const;
  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  unsigned get_n_active_sms() const;
  opu_sim *get_gpu() { return m_opu; }

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  //void get_cache_stats(cache_stats &cs) const;
  //void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  //void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  //void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  //void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  //void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  virtual void create_shader_core_ctx() = 0;

  shader_core_ctx *get_core(int id_in_cluster) { return m_core[id_in_cluster]; }
 protected:
  unsigned m_cluster_id;
  opu_sim *m_opu;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  shader_core_ctx **m_core;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;
};

class exec_simt_core_cluster : public simt_core_cluster {
 public:
  exec_simt_core_cluster(class opu_sim *gpu, unsigned cluster_id,
                         const shader_core_config *config,
                         class shader_core_stats *stats)
      : simt_core_cluster(gpu, cluster_id, config, stats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

