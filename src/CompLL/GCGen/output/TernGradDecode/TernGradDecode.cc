
#include "TernGradDecode_wrapper.h"
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

namespace mxnet{
namespace op{
DMLC_REGISTER_PARAMETER(TernGradDecode_param);

NNVM_REGISTER_OP(_contrib_TernGradDecode)
.set_attr_parser(ParamParser<TernGradDecode_param>)
.set_num_inputs(2)
.set_num_outputs(0)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs){
        return std::vector<std::string>{ "compressed","gradient" };
}
)
.set_attr<mxnet::FInferShape>("FInferShape", TernGradDecode_shape)
.set_attr<nnvm::FInferType>("FInferType", TernGradDecode_type)
.add_argument("compressed", "NDArray", "array")
.add_argument("gradient", "NDArray", "array")
.add_arguments(TernGradDecode_param::__FIELDS__())
;

NNVM_REGISTER_OP(_contrib_TernGradDecode)
.set_attr<FCompute>("FCompute<cpu>", TernGradDecode<cpu, thrust::detail::host_t>)
;


}
}
