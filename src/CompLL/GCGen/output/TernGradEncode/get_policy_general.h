
#ifndef ZQ_CPP_LIB_OPERATE_MEMORY_GET_POLICY_INL_H
#define ZQ_CPP_LIB_OPERATE_MEMORY_GET_POLICY_INL_H

#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h> //



namespace zq_cpp_lib {
namespace operate_memory {




template <typename T>
struct get_policy{};


template<>
struct get_policy<thrust::system::omp::detail::par_t>{
  inline static thrust::system::omp::detail::par_t get(
    void* stream
  ){
    return thrust::omp::par;
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyDoubleIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void streamSynchronize(
    void* stream
  ){
    return;
  }
};

template<>
struct get_policy<thrust::detail::host_t>{
  inline static thrust::detail::host_t get(
    void* stream
  ){
    return thrust::host;
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyDoubleIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void streamSynchronize(
    void* stream
  ){
    return;
  }
};

#define PVOID2CUDASTREAM(stream) (static_cast<cudaStream_t>(stream))

template<>
struct get_policy<thrust::cuda_cub::par_t::stream_attachment_type>{
  inline static thrust::cuda_cub::par_t::stream_attachment_type get(
    void* stream
  ){
    return thrust::cuda::par.on(PVOID2CUDASTREAM(stream));
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    void* stream  //pointer to stream
  ){
    cudaMemcpyAsync(dst,src,len,cudaMemcpyDeviceToHost, PVOID2CUDASTREAM(stream));
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    cudaMemcpyAsync(dst,src,len,cudaMemcpyHostToDevice,PVOID2CUDASTREAM(stream));
  }  
  inline static void memcpyDoubleIn(
    void* dst,
    void* src,
    uint32_t len,
    void* stream
  ){
    cudaMemcpyAsync(dst,src,len,cudaMemcpyDeviceToDevice,PVOID2CUDASTREAM(stream));
  }
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    cudaMemcpy(dst,src,len,cudaMemcpyDeviceToHost);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    cudaMemcpy(dst,src,len,cudaMemcpyHostToDevice);
  }
  inline static void streamSynchronize(
    void* stream
  ){
    cudaStreamSynchronize(PVOID2CUDASTREAM(stream));
  }
};

}   
}   

#endif
