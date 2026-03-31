/**
 * SlipStream — Thread Pool Implementation
 *
 * A minimal pthreads-based work queue for background prefetching.
 */

#include "utils/thread_pool.h"

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// ─── Task Queue ──────────────────────────────────────────────────────────────

#define SS_MAX_QUEUE 256

typedef struct {
    ss_task_func_t func;
    void          *arg;
} ss_task_t;

struct ss_thread_pool {
    pthread_t       *threads;
    uint32_t         n_threads;

    ss_task_t        queue[SS_MAX_QUEUE];
    int32_t          queue_head;
    int32_t          queue_tail;
    int32_t          queue_count;

    pthread_mutex_t  mutex;
    pthread_cond_t   cond_work;     // Signal: work available
    pthread_cond_t   cond_done;     // Signal: all work done

    int32_t          active_count;  // Number of threads processing tasks
    bool             shutdown;
};

// ─── Worker Thread ───────────────────────────────────────────────────────────

static void *worker_thread(void *arg) {
    ss_thread_pool_t *pool = (ss_thread_pool_t *)arg;

    while (true) {
        pthread_mutex_lock(&pool->mutex);

        // Wait for work
        while (pool->queue_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }

        if (pool->shutdown && pool->queue_count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // Dequeue task
        ss_task_t task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % SS_MAX_QUEUE;
        pool->queue_count--;
        pool->active_count++;

        pthread_mutex_unlock(&pool->mutex);

        // Execute
        task.func(task.arg);

        // Mark done
        pthread_mutex_lock(&pool->mutex);
        pool->active_count--;
        if (pool->queue_count == 0 && pool->active_count == 0) {
            pthread_cond_signal(&pool->cond_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }

    return NULL;
}

// ─── API ─────────────────────────────────────────────────────────────────────

uint32_t ss_get_cpu_count(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
#else
    return 1;
#endif
}

ss_thread_pool_t *ss_thread_pool_create(uint32_t n_threads) {
    if (n_threads == 0) n_threads = ss_get_cpu_count();
    if (n_threads > 32) n_threads = 32;

    ss_thread_pool_t *pool = (ss_thread_pool_t *)calloc(1, sizeof(ss_thread_pool_t));
    if (!pool) return NULL;

    pool->n_threads = n_threads;
    pool->shutdown = false;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_work, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    pool->threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    if (!pool->threads) {
        free(pool);
        return NULL;
    }

    for (uint32_t i = 0; i < n_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }

    return pool;
}

bool ss_thread_pool_submit(ss_thread_pool_t *pool, ss_task_func_t func, void *arg) {
    if (!pool || !func) return false;

    pthread_mutex_lock(&pool->mutex);

    if (pool->queue_count >= SS_MAX_QUEUE) {
        pthread_mutex_unlock(&pool->mutex);
        return false;
    }

    pool->queue[pool->queue_tail].func = func;
    pool->queue[pool->queue_tail].arg = arg;
    pool->queue_tail = (pool->queue_tail + 1) % SS_MAX_QUEUE;
    pool->queue_count++;

    pthread_cond_signal(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);

    return true;
}

void ss_thread_pool_wait(ss_thread_pool_t *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    while (pool->queue_count > 0 || pool->active_count > 0) {
        pthread_cond_wait(&pool->cond_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

void ss_thread_pool_destroy(ss_thread_pool_t *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);

    for (uint32_t i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_work);
    pthread_cond_destroy(&pool->cond_done);

    free(pool->threads);
    free(pool);
}
