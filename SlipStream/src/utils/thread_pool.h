/**
 * SlipStream — Thread Pool
 */

#ifndef SS_THREAD_POOL_H
#define SS_THREAD_POOL_H

#include <stdint.h>
#include <stdbool.h>

typedef struct ss_thread_pool ss_thread_pool_t;
typedef void (*ss_task_func_t)(void *arg);

/**
 * Create a thread pool with the given number of threads.
 * If n_threads is 0, uses the system's CPU count.
 */
ss_thread_pool_t *ss_thread_pool_create(uint32_t n_threads);

/**
 * Submit a task to the thread pool.
 */
bool ss_thread_pool_submit(ss_thread_pool_t *pool, ss_task_func_t func, void *arg);

/**
 * Wait for all submitted tasks to complete.
 */
void ss_thread_pool_wait(ss_thread_pool_t *pool);

/**
 * Destroy the thread pool. Waits for pending tasks to finish.
 */
void ss_thread_pool_destroy(ss_thread_pool_t *pool);

/**
 * Get the number of available CPU cores.
 */
uint32_t ss_get_cpu_count(void);

#endif // SS_THREAD_POOL_H
