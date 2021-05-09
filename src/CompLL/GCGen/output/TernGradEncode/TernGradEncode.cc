
#include "TernGradEncode_wrapper.h"
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace mxnet{
namespace op{
DMLC_REGISTER_PARAMETER(TernGradEncode_param);

NNVM_REGISTER_OP(_contrib_TernGradEncode)
.set_attr_parser(ParamParser<TernGradEncode_param>)
.set_num_inputs(2)
.set_num_outputs(0)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs){
        return std::vector<std::string>{ "gradient","compressed" };
}
)
.set_attr<mxnet::FInferShape>("FInferShape", TernGradEncode_shape)
.set_attr<nnvm::FInferType>("FInferType", TernGradEncode_type)
.add_argument("gradient", "NDArray", "array")
.add_argument("compressed", "NDArray", "array")
.add_arguments(TernGradEncode_param::__FIELDS__())
;

NNVM_REGISTER_OP(_contrib_TernGradEncode)
.set_attr<FCompute>("FCompute<cpu>", TernGradEncode<cpu, thrust::detail::host_t>)
;


}
}
