#ifndef GET_STREAM_MXNET_H
#define GET_STREAM_MXNET_H

#include <mxnet/operator_util.h>

#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace mxnet{
    namespace op{
        template <typename T>
        struct get_stream{
            inline static void* get(const OpContext& ctx){
                return NULL;
            }
        };

        template<>
        struct get_stream<thrust::cuda_cub::par_t::stream_attachment_type>{
            inline static void* get(const OpContext& ctx){
                auto stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
                return static_cast<void*>(stream);
            }
        };
    }
}

#endif
