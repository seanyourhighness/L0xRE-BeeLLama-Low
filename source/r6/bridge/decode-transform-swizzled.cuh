__global__ void decode_transform_swizzled(const float *x, const half *rin, const float *sin, half *dst, int ic, int m) {
 const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
 const int tile=blockIdx.x*(blockDim.x/32)+warp, chunks=ic/128;
 if(tile>=m*chunks)return;
 const int row=tile/chunks, group=tile%chunks, k=group*128+lane*4;
 float v[4];
 #pragma unroll
 for(int j=0;j<4;j++) {
  float a=__half2float(__float2half_rn(x[row*ic+k+j]));
  v[j]=decode_ftz_mul(decode_ftz_mul(a,sin[k+j]),__half2float(rin[k+j]));
 }
 float a=decode_ftz_add(v[0],v[1]),b=decode_ftz_sub(v[0],v[1]);
 float c=decode_ftz_add(v[2],v[3]),d=decode_ftz_sub(v[2],v[3]);
 v[0]=decode_ftz_add(a,c);v[1]=decode_ftz_add(b,d);v[2]=decode_ftz_sub(a,c);v[3]=decode_ftz_sub(b,d);
 #pragma unroll
 for(int mask=1;mask<=16;mask*=2) {
  float sign=(lane&mask)?-1.f:1.f;
  #pragma unroll
  for(int j=0;j<4;j++) {float other=__shfl_xor_sync(0xffffffff,v[j],mask);v[j]=decode_ftz_fma(v[j],sign,other);}
 }
 #pragma unroll
 for(int j=0;j<4;j++)dst[row*ic+((k+j)&~15)+(((k+j)&6)<<1)+(((k+j)&8)>>2)+((k+j)&1)]=__float2half_rn(decode_ftz_mul(v[j],__uint_as_float(0x3db504f3)));
}
