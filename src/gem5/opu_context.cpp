
typedef IsaSim* (*pfn_make_isasim)(gpgpu_t* gpu, gpgpu_context *ctx);

IsaSim* gpgpu_context::get_isasim() {
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
#if 0
void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
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
  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_the_gpu;
}
#endif

gpgpu_sim *OpuContext::gem5_ptx_sim_init_perf(stream_manager **p_stream_manager, CudaGPU *cuda_gpu, const char *config_path)
{
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
      new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this, cuda_gpu);
  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  *p_stream_manager = the_gpgpusim->g_stream_manager;

  return the_gpgpusim->g_the_gpu;
}


