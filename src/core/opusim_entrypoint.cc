#include "opusim_entrypoint.h"
#include <stdio.h>

// #include "../libcuda_sim/gpgpu_context.h"
#include "opu-sim.h"
#include "option_parser.h"
#include "stream_manager.h"
#include <dlfcn.h>
//#include "opu/isasim/executor/inc/IsaSim.h"
class IsaSim;

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  // at most one kernel running at a time
  bool done;
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    done = true;
    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      done = false;
      ctx->the_gpgpusim->g_the_gpu->init();
      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        // TODO schi
        // ctx->the_gpgpusim->g_the_gpu->cycle();
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
  } while (!done);
  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

static void termination_callback() {
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);
}

void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  atexit(termination_callback);
  // concurrent kernel execution simulation thread
  do {
    if (g_debug_execution >= 3) {
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);
    }
    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done)
      ;
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      fflush(stdout);
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = true;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    bool active = false;
    bool sim_cycles = false;
    ctx->the_gpgpusim->g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
          !ctx->the_gpgpusim->g_the_gpu->active())
        break;

      // functional simulation
      /*
      if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
        kernel_info_t *kernel =
            ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
        assert(kernel);
        ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
            *kernel);
        ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
      }
      */

      // performance simulation
      if (ctx->the_gpgpusim->g_the_gpu->active()) {
        // TODO schi
        // ctx->the_gpgpusim->g_the_gpu->cycle();
        sim_cycles = true;
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      /* else {
        if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          ctx->the_gpgpusim->g_sim_done = true;
          ctx->the_gpgpusim->break_limit = true;
        }
      }
      */

      active = ctx->the_gpgpusim->g_the_gpu->active() ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());

    } while (active && !ctx->the_gpgpusim->g_sim_done);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }
    if (sim_cycles) {
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
  } while (!ctx->the_gpgpusim->g_sim_done);

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}



//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true, the flag used by the function to distinguish OpenCL calls from
  // the CUDA simulation calls which it is needed by the called function to not
  // register the exit the exit of OpenCL kernel as it doesn't register entering
  // in the first place as the CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  return 0;
}
