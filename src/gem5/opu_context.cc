#include "opu_context.hh"
#include "opu_stream.hh"
#include "../opuusim_base.h"
#include "../opusim_config.h"
#include <cstdio>
#include <dlfcn.h>

opu_sim_config *g_config {nullptr};

namespace gem5 {
OpuSimBase *g_the_opu {nullptr};


#if 0
typedef IsaSim* (*pfn_make_isasim)(gpgpu_t* gpu, OpuContext *ctx);

IsaSim* OpuContext::get_isasim() {
    static IsaSim* isasim = nullptr;
    if (isasim == nullptr) {
        void* lib_handle = dlopen("libisasim.so", RTLD_NOW | RTLD_GLOBAL);
        if (lib_handle == nullptr) {
            printf("Failed to load libisasim.so, error - %sn\n", dlerror());
            exit(-1);
        }
        pfn_make_isasim make_isasim = (pfn_make_isasim)dlsym(lib_handle, "make_isasim");

        if (make_isasim == nullptr) {
            printf("Failed to dlsym make_isasim, error - %sn\n", dlerror());
            exit(-1);
        }
        isasim = make_isasim(this->the_gpgpusim->g_the_gpu, this);
    }
    return isasim;
}
void OpuContext::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_OpuStream->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_OpuStream->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

void OpuContext::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

gpgpu_sim *OpuContext::gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  func_sim->read_sim_environment_variables();
  ptx_parser->read_parser_environment_variables();
  option_parser_t opp = option_parser_create();

  ptx_reg_options(opp);
  func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);
  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  the_gpgpusim->g_the_gpu_config->init();

  the_gpgpusim->g_the_gpu =
      new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
  the_gpgpusim->g_OpuStream = new OpuStream(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_the_gpu;
}
#endif

typedef OpuSimBase* (*pfn_make_opusim)(opu_sim_config* config, OpuContext *ctx, gem5::OpuTop *);

OpuSimBase *OpuContext::gem5_opu_sim_init(OpuStream **p_opu_stream, gem5::OpuTop *opu_top, const char *config_path)
{
  // the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  if (g_the_opu == nullptr) {
    void* lib_handle = dlopen("libopusim.so", RTLD_NOW | RTLD_GLOBAL);
    if (lib_handle == nullptr) {
        printf("Failed to load libopusim.so, error - %sn\n", dlerror());
        exit(-1);
    }
    pfn_make_opusim make_opusim = (pfn_make_opusim)dlsym(lib_handle, "make_opusim");

    if (make_opusim == nullptr) {
        printf("Failed to dlsym make_opusim, error - %sn\n", dlerror());
        exit(-1);
    }
    //g_config = new opu_sim_config();
    g_the_opu = make_opusim(g_config, this, opu_top);
  }

  m_opu_stream = new OpuStream(g_the_opu, true);

  *p_opu_stream = m_opu_stream;

  return g_the_opu;
}


}
