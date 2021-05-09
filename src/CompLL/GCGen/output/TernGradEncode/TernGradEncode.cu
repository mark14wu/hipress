
#include "TernGradEncode_wrapper.h"
#include <thrust/execution_policy.h>

namespace mxnet{
namespace op{

NNVM_REGISTER_OP(_contrib_TernGradEncode)
.set_attr<FCompute>("FCompute<gpu>", TernGradEncode<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}
}
