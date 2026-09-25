// Minimal dynamic-loader shim shared by the Escha bridge consumers.
// POSIX: dlopen/dlsym/dlerror.  Windows: LoadLibraryA/GetProcAddress/GetLastError.
#pragma once

#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

inline void * escha_dl_open(const char * path) {
    return (void *) LoadLibraryA(path);
}
inline void * escha_dl_sym(void * handle, const char * name) {
    return (void *) GetProcAddress((HMODULE) handle, name);
}
inline const char * escha_dl_error() {
    static char buf[64];
    std::snprintf(buf, sizeof(buf), "LoadLibrary/GetProcAddress failed (Win32 error %lu)",
                  (unsigned long) GetLastError());
    return buf;
}
#else
#include <dlfcn.h>
inline void * escha_dl_open(const char * path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
inline void * escha_dl_sym(void * handle, const char * name) {
    return dlsym(handle, name);
}
inline const char * escha_dl_error() {
    const char * e = dlerror();
    return e ? e : "no dl error";
}
#endif
