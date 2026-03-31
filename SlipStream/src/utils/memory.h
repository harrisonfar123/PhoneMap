/**
 * SlipStream — Memory Utilities
 */

#ifndef SS_MEMORY_H
#define SS_MEMORY_H

#include <stddef.h>
#include <stdint.h>

/**
 * Simple arena allocator for temporary per-inference allocations.
 * Avoids malloc/free overhead during hot loops.
 */
typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   offset;
} ss_arena_t;

ss_arena_t *ss_arena_create(size_t size);
void       *ss_arena_alloc(ss_arena_t *arena, size_t size);
void        ss_arena_reset(ss_arena_t *arena);
void        ss_arena_free(ss_arena_t *arena);

/**
 * Get system physical memory in bytes.
 */
uint64_t ss_get_physical_memory(void);

/**
 * Get available (free) memory in bytes.
 */
uint64_t ss_get_available_memory(void);

#endif // SS_MEMORY_H
