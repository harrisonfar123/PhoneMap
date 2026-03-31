/**
 * SlipStream — Memory Utilities Implementation
 */

#include "utils/memory.h"
#include <slipstream.h>

#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

// ─── Arena Allocator ─────────────────────────────────────────────────────────

ss_arena_t *ss_arena_create(size_t size) {
    ss_arena_t *arena = (ss_arena_t *)malloc(sizeof(ss_arena_t));
    if (!arena) return NULL;

    arena->base = (uint8_t *)malloc(size);
    if (!arena->base) {
        free(arena);
        return NULL;
    }

    arena->size = size;
    arena->offset = 0;
    return arena;
}

void *ss_arena_alloc(ss_arena_t *arena, size_t size) {
    if (!arena) return NULL;

    // Align to 16 bytes
    size_t aligned_offset = (arena->offset + 15) & ~(size_t)15;

    if (aligned_offset + size > arena->size) {
        return NULL;  // Arena full
    }

    void *ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

void ss_arena_reset(ss_arena_t *arena) {
    if (arena) arena->offset = 0;
}

void ss_arena_free(ss_arena_t *arena) {
    if (!arena) return;
    free(arena->base);
    free(arena);
}

// ─── System Memory ───────────────────────────────────────────────────────────

uint64_t ss_get_physical_memory(void) {
#ifdef __APPLE__
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    sysctl(mib, 2, &mem, &len, NULL, 0);
    return mem;
#elif defined(__linux__)
    struct sysinfo si;
    sysinfo(&si);
    return (uint64_t)si.totalram * si.mem_unit;
#else
    return 0;
#endif
}

uint64_t ss_get_available_memory(void) {
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t page_size = 4096;
        return (uint64_t)(vm_stat.free_count + vm_stat.inactive_count) * page_size;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo si;
    sysinfo(&si);
    return (uint64_t)si.freeram * si.mem_unit;
#else
    return 0;
#endif
}

uint64_t ss_get_memory_usage(void) {
#ifdef __APPLE__
    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vmInfo, &count);
    if (kr == KERN_SUCCESS) {
        return (uint64_t)vmInfo.phys_footprint;
    }
    return 0;
#else
    return 0;
#endif
}
