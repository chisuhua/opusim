#include "cache_stats.h"
#include "mem_common.h"
#include <cassert>

cache_stats::cache_stats() {
  m_stats.resize(NUM_MEM_ACCESS_TYPE);
  m_stats_pw.resize(NUM_MEM_ACCESS_TYPE);
  m_fail_stats.resize(NUM_MEM_ACCESS_TYPE);
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_stats_pw[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_fail_stats[i].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
    std::fill(m_fail_stats[i].begin(), m_fail_stats[i].end(), 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
  }
}

#if 0
void cache_stats::inc_stats(int access_type, int access_outcome) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  m_stats[access_type][access_outcome]++;
}
#endif

void cache_stats::inc_stats_pw(int access_type, int access_outcome) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");
  m_stats_pw[access_type][access_outcome]++;
}

#if 0
void cache_stats::inc_fail_stats(int access_type, int fail_outcome) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  m_fail_stats[access_type][fail_outcome]++;
}
#endif

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  cache_stats ret;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      ret(type, status, false) =
          m_stats[type][status] + cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      ret(type, status, true) =
          m_fail_stats[type][status] + cs(type, status, true);
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats_pw[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      m_fail_stats[type][status] += cs(type, status, true);
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const {
#if 0
  ///
  /// Print out each non-zero cache statistic for every memory access type and
  /// status "cache_name" defaults to "Cache_stats" when no argument is
  /// provided, otherwise the provided name is used. The printed format is
  /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  ///
  std::vector<unsigned> total_access;
  total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_request_status_str((enum cache_request_status)status),
              m_stats[type][status]);

      if (status != RESERVATION_FAIL && status != MSHR_HIT)
        // MSHR_HIT is a special type of SECTOR_MISS
        // so its already included in the SECTOR_MISS
        total_access[type] += m_stats[type][status];
    }
  }
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    if (total_access[type] > 0)
      fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
              total_access[type]);
  }
#endif
}

void cache_stats::print_fail_stats(FILE *fout, const char *cache_name) const {
#if 0
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
      if (m_fail_stats[type][fail] > 0) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                m_fail_stats[type][fail]);
      }
    }
  }
#endif
}

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0;
  for (unsigned type = 0; type < num_access_type; ++type) {
    for (unsigned status = 0; status < num_access_status; ++status) {
      if (!check_valid((int)access_type[type], (int)access_status[status]))
        assert(0 && "Unknown cache access type or access outcome");
      total += m_stats[access_type[type]][access_status[status]];
    }
  }
  return total;
}


bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}
bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

