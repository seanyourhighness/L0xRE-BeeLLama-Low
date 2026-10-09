#pragma once
#include <cstdlib>
#include <string>

// Experimental process-start feature configuration. Call sites deliberately
// exclude candidate attestation and diagnostic/test controls. Each site owns a
// copy, preserving missing versus empty values and avoiding CRT pointer lifetime
// hazards. Changing a snapshotted feature requires a fresh process.
struct l0xre_startup_env_value {
    bool present;
    std::string value;
    explicit l0xre_startup_env_value(const char * name) {
        const char * raw = std::getenv(name);
        present = raw != nullptr;
        if (present) value = raw;
    }
    const char * get() const { return present ? value.c_str() : nullptr; }
};

#define L0XRE_STARTUP_ENV(name) ([]() -> const char * { \
    static const l0xre_startup_env_value snapshot(name); \
    return snapshot.get(); \
}())
