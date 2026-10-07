#include <windows.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
using prefill_fn = int (*)(void *, int, const float *, const float *, const float *,
    const float *, const float *, const float *, float *, float *,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,int64_t);
using error_fn = const char *(*)();
int main() {
    HMODULE lib=LoadLibraryA("escha_r6_gdn_masked.dll");
    if (!lib) { std::fprintf(stderr,"LoadLibrary failed: %lu\n",GetLastError()); return 1; }
    auto prefill=reinterpret_cast<prefill_fn>(GetProcAddress(lib,"escha_gdn_chunk_prefill_masked_v1"));
    auto error=reinterpret_cast<error_fn>(GetProcAddress(lib,"escha_gdn_chunk_error"));
    if (!prefill || !error) return 2;
    int null_status=prefill(nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                           128,48,512,1,128,6144,128,6144,1,48,1.0f/std::sqrt(128.0f),512);
    float input=3.0f,output=12345.0f,state=67890.0f;
    int device_status=prefill(nullptr,1,&input,&input,&input,&input,&input,&input,&output,&state,
                             128,48,512,1,128,6144,128,6144,1,48,1.0f/std::sqrt(128.0f),512);
    bool pass=null_status==1 && device_status==1 && output==12345.0f && state==67890.0f
              && error()!=nullptr && std::strlen(error())==0;
    std::printf("{\"status\":\"%s\",\"gpu_work_launched\":false,\"null_argument_status\":%d,"
                "\"unsupported_device_status\":%d,\"output_guards_preserved\":%s}\n",
                pass?"pass":"failed",null_status,device_status,
                output==12345.0f&&state==67890.0f?"true":"false");
    FreeLibrary(lib);
    return pass?0:3;
}
