#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <dlfcn.h>
using Fn=int(*)(cudaStream_t,int,void*,const void*,const int16_t*,const void*,int,int,int,int,int);
using Error=const char*(*)();
struct Bridge{void *handle;Fn run;Error error;};
static Bridge open(const char *p){void*h=dlopen(p,RTLD_NOW|RTLD_LOCAL);if(!h){fprintf(stderr,"%s\n",dlerror());exit(2);}return{h,(Fn)dlsym(h,"escha_official_code_gemm_pretransformed"),(Error)dlsym(h,"escha_official_bridge_error")};}
static void ck(cudaError_t e,const char*p){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",p,cudaGetErrorString(e));exit(3);}}
static uint32_t next(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static uint16_t bits(float v){half h=__float2half_rn(v);uint16_t b;memcpy(&b,&h,2);return b;}
struct Data{int ic,oc,k,acc,role;int16_t*code=nullptr;uint16_t*x=nullptr,*rout=nullptr,*a=nullptr,*b=nullptr;std::vector<uint16_t> input,ha,hb;};
int main(int argc,char**argv){
 if(argc!=4)return 2;
 Bridge integer=open(argv[1]),candidate=open(argv[2]),native=open(argv[3]);
 int mask=atoi(getenv("L0XRE_INT8_PROJ_MASK")),mode=atoi(getenv("L0XRE_INT8_PREFILL"));
 cudaStream_t stream,other;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking),"other");
 std::vector<Data> data={{5120,17408,2,1,1},{5120,17408,3,1,2},{17408,5120,3,0,0},{5120,10240,2,1,3},{6144,5120,2,1,4},{5120,6144,2,1,5},{5120,12288,2,1,6},{5120,1024,2,1,7},{5120,5120,2,1,-1}};
 uint32_t seed=0x53b55055;
 for(auto &d:data){
  size_t ncode=size_t(d.ic)*d.oc*d.k/16;
  std::vector<uint16_t> code(ncode),rout(d.oc,bits(.5f));for(auto &w:code)w=uint16_t(next(seed));
  d.input.resize(size_t(512)*d.ic);d.ha.resize(size_t(512)*d.oc);d.hb.resize(d.ha.size());
  for(auto&w:d.input)w=bits(float(int(next(seed)%2001)-1000)/100000.f);
  ck(cudaMalloc(&d.code,ncode*2),"code alloc");ck(cudaMalloc(&d.x,d.input.size()*2),"input alloc");ck(cudaMalloc(&d.rout,d.oc*2),"rout alloc");
  ck(cudaMalloc(&d.a,d.ha.size()*2),"reference alloc");ck(cudaMalloc(&d.b,d.hb.size()*2),"candidate alloc");
  ck(cudaMemcpy(d.code,code.data(),ncode*2,cudaMemcpyHostToDevice),"code copy");ck(cudaMemcpy(d.rout,rout.data(),d.oc*2,cudaMemcpyHostToDevice),"rout copy");ck(cudaMemcpy(d.x,d.input.data(),d.input.size()*2,cudaMemcpyHostToDevice),"input copy");
 }
 auto call=[&](Bridge &b,Data &d,cudaStream_t s,uint16_t*out,int m){int rc=b.run(s,0,out,d.x,d.code,d.rout,m,d.ic,d.oc,d.k,d.acc);if(rc){fprintf(stderr,"bridge rc%d: %s\n",rc,b.error());exit(4);}};
 auto expected=[&](Data&d)->Bridge&{return mode==1&&d.role>=0&&(mask&(1<<d.role))?integer:native;};
 auto compare=[&](Data&d,int m,const char*label){ck(cudaDeviceSynchronize(),"compare sync");ck(cudaMemcpy(d.ha.data(),d.a,size_t(m)*d.oc*2,cudaMemcpyDeviceToHost),"ref copy");ck(cudaMemcpy(d.hb.data(),d.b,size_t(m)*d.oc*2,cudaMemcpyDeviceToHost),"candidate copy");size_t n=0;for(size_t i=0;i<size_t(m)*d.oc;i++)n+=d.ha[i]!=d.hb[i];fprintf(stderr,"CHECK %s role=%d M=%d unequal=%zu\n",label,d.role,m,n);if(n)exit(5);};
 for(int m:{256,512})for(auto&d:data){call(expected(d),d,nullptr,d.a,m);ck(cudaDeviceSynchronize(),"reference sync");call(candidate,d,stream,d.b,m);compare(d,m,"warmup");}
 cudaGraph_t graph;cudaGraphExec_t exec;
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"begin capture");
 for(int repeat=0;repeat<2;repeat++)for(auto&d:data)call(candidate,d,stream,d.b,512);
 ck(cudaStreamEndCapture(stream,&graph),"end capture");ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
 for(int replay=0;replay<3;replay++){
  for(auto&d:data){for(auto&w:d.input)w=bits(float(int(next(seed)%2001)-1000)/100000.f);ck(cudaMemcpy(d.x,d.input.data(),d.input.size()*2,cudaMemcpyHostToDevice),"change input");call(expected(d),d,nullptr,d.a,512);ck(cudaDeviceSynchronize(),"reference sync");}
  ck(cudaGraphLaunch(exec,stream),"replay");for(auto&d:data)compare(d,512,"mixed graph replay");
 }
 for(auto&d:data){call(native,d,nullptr,d.a,512);ck(cudaDeviceSynchronize(),"native sync");call(candidate,d,other,d.b,512);compare(d,512,"other-stream fallback");call(native,d,nullptr,d.a,4);ck(cudaDeviceSynchronize(),"native small sync");call(candidate,d,stream,d.b,4);compare(d,4,"small-batch bypass");}
 ck(cudaGraphExecDestroy(exec),"destroy graph exec");ck(cudaGraphDestroy(graph),"destroy graph");
 printf("{\"passed\":true,\"mask\":%d,\"mode\":%d,\"mixed_shape_replays\":3,\"shape_count\":9}\n",mask,mode);
 return 0;
}
