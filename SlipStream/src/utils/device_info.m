/**
 * SlipStream — Device Hardware Detection
 *
 * Queries Apple-specific system information including device model,
 * chip name, core counts, memory, GPU info, and thermal state.
 */

#import <Foundation/Foundation.h>
#import <sys/sysctl.h>
#import <mach/mach.h>

#ifdef __APPLE__
#import <os/proc.h>
#endif

#ifdef SS_HAS_METAL
#import <Metal/Metal.h>
#endif

#include <slipstream.h>

// ─── Internal helpers ────────────────────────────────────────────────────────

static void sysctl_string(const char *name, char *buf, size_t buflen) {
    size_t len = buflen;
    if (sysctlbyname(name, buf, &len, NULL, 0) != 0) {
        buf[0] = '\0';
    }
}

static uint32_t sysctl_uint32(const char *name) {
    uint32_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, NULL, 0);
    return val;
}

static uint64_t sysctl_uint64(const char *name) {
    uint64_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, NULL, 0);
    return val;
}

// Map hw.machine codes to human-readable chip names.
// This covers the most common iPhone/iPad chips.
static const char *chip_name_for_model(const char *model) {
    if (!model) return "Unknown";
    
    // iPhone identifiers
    if (strstr(model, "iPhone17"))  return "Apple A18 Pro";
    if (strstr(model, "iPhone16"))  return "Apple A17 Pro";
    if (strstr(model, "iPhone15"))  return "Apple A16 Bionic";
    if (strstr(model, "iPhone14"))  return "Apple A15 Bionic";
    if (strstr(model, "iPhone13"))  return "Apple A15 Bionic";
    if (strstr(model, "iPhone12"))  return "Apple A14 Bionic";
    if (strstr(model, "iPhone11"))  return "Apple A13 Bionic";
    if (strstr(model, "iPhone10"))  return "Apple A11 Bionic";
    
    // iPad identifiers (common)
    if (strstr(model, "iPad16"))    return "Apple M4";
    if (strstr(model, "iPad14"))    return "Apple M2";
    if (strstr(model, "iPad13"))    return "Apple M1";
    
    // Apple Silicon Macs (for simulator / macOS builds)
    if (strstr(model, "arm64"))     return "Apple Silicon";
    
    // x86 simulator
    if (strstr(model, "x86_64"))    return "Intel (Simulator)";
    
    return "Unknown Apple SoC";
}

// ─── Public API ──────────────────────────────────────────────────────────────

ss_error_t ss_get_device_info(ss_device_info_t *info) {
    if (!info) return SS_ERROR_INVALID_PARAMS;
    
    memset(info, 0, sizeof(ss_device_info_t));
    
    // Device model identifier (e.g. "iPhone15,2")
    sysctl_string("hw.machine", info->device_model, sizeof(info->device_model));
    
    // If hw.machine returns empty (macOS), try hw.model
    if (info->device_model[0] == '\0') {
        sysctl_string("hw.model", info->device_model, sizeof(info->device_model));
    }
    
    // Chip name from model lookup
    const char *chip = chip_name_for_model(info->device_model);
    strncpy(info->chip_name, chip, sizeof(info->chip_name) - 1);
    
    // CPU core counts
    info->cpu_core_count = sysctl_uint32("hw.ncpu");
    
    // Try to get performance/efficiency core split (macOS 12+ / iOS 16+)
    info->perf_core_count = sysctl_uint32("hw.perflevel0.logicalcpu");
    info->efficiency_core_count = sysctl_uint32("hw.perflevel1.logicalcpu");
    
    // If perflevel sysctls aren't available, fall back
    if (info->perf_core_count == 0 && info->efficiency_core_count == 0) {
        info->perf_core_count = info->cpu_core_count;
    }
    
    // Memory
    info->total_memory = sysctl_uint64("hw.memsize");
    
#if defined(__APPLE__) && TARGET_OS_IOS
    info->available_memory = (uint64_t)os_proc_available_memory();
#else
    // macOS fallback via vm_statistics
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t page_size = 4096;
        info->available_memory = (uint64_t)(vm_stat.free_count + vm_stat.inactive_count) * page_size;
    }
#endif
    
    // GPU core count via Metal
#ifdef SS_HAS_METAL
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            // Metal doesn't expose GPU core count directly.
            // We can infer from recommended thread group sizes as a rough proxy.
            // For now, store 0 and let the Swift layer handle display.
            info->gpu_core_count = 0;
        }
    }
#endif
    
    // Thermal state via NSProcessInfo
    @autoreleasepool {
        NSProcessInfo *proc = [NSProcessInfo processInfo];
        info->thermal_state = (int32_t)proc.thermalState;
    }
    
    return SS_OK;
}

// ─── CPU Usage ───────────────────────────────────────────────────────────────

double ss_get_cpu_usage(void) {
    thread_array_t threads;
    mach_msg_type_number_t thread_count;
    
    kern_return_t kr = task_threads(mach_task_self(), &threads, &thread_count);
    if (kr != KERN_SUCCESS) return -1.0;
    
    double total_usage = 0.0;
    
    for (mach_msg_type_number_t i = 0; i < thread_count; i++) {
        thread_basic_info_data_t info;
        mach_msg_type_number_t info_count = THREAD_BASIC_INFO_COUNT;
        
        kr = thread_info(threads[i], THREAD_BASIC_INFO,
                         (thread_info_t)&info, &info_count);
        if (kr == KERN_SUCCESS && !(info.flags & TH_FLAGS_IDLE)) {
            total_usage += (double)info.cpu_usage / TH_USAGE_SCALE * 100.0;
        }
        
        mach_port_deallocate(mach_task_self(), threads[i]);
    }
    
    vm_deallocate(mach_task_self(), (vm_address_t)threads,
                  thread_count * sizeof(thread_t));
    
    return total_usage;
}
