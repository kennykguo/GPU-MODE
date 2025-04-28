#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel_1.fatbin.c"
extern void __device_stub__Z11sgemm_naiveiiifPfPKffS_(int, int, int, float, float *, const float *, float, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z11sgemm_naiveiiifPfPKffS_(int __par0, int __par1, int __par2, float __par3, float *__par4, const float *__par5, float __par6, float *__par7){__cudaLaunchPrologue(8);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__cudaSetupArgSimple(__par2, 8UL);__cudaSetupArgSimple(__par3, 12UL);__cudaSetupArgSimple(__par4, 16UL);__cudaSetupArgSimple(__par5, 24UL);__cudaSetupArgSimple(__par6, 32UL);__cudaSetupArgSimple(__par7, 40UL);__cudaLaunch(((char *)((void ( *)(int, int, int, float, float *, const float *, float, float *))sgemm_naive)));}
# 10 "kernel_1.cu"
void sgemm_naive( int __cuda_0,int __cuda_1,int __cuda_2,float __cuda_3,float *__cuda_4,const float *__cuda_5,float __cuda_6,float *__cuda_7)
# 10 "kernel_1.cu"
{__device_stub__Z11sgemm_naiveiiifPfPKffS_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 21 "kernel_1.cu"
}
# 1 "kernel_1.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T9) {  __nv_dummy_param_ref(__T9); __nv_save_fatbinhandle_for_managed_rt(__T9); __cudaRegisterEntry(__T9, ((void ( *)(int, int, int, float, float *, const float *, float, float *))sgemm_naive), _Z11sgemm_naiveiiifPfPKffS_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
