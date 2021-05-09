
#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

#include "TernGradEncode_body.h"
#include "get_stream_mxnet.h"


#define uint8 uint8_t
#define uint32 uint32_t
#define int32 int32_t


namespace mxnet{
namespace op{
struct TernGradEncode_param : public dmlc::Parameter<TernGradEncode_param> {
	uint8 bitwidth;

DMLC_DECLARE_PARAMETER(TernGradEncode_param){
	DMLC_DECLARE_FIELD(bitwidth).describe("Describe something for this parameter");

};

};

inline bool TernGradEncode_shape(const nnvm::NodeAttrs& attrs, mxnet::ShapeVector* in_attrs, mxnet::ShapeVector* out_attrs){
    return true;
};

inline bool TernGradEncode_type(const nnvm::NodeAttrs& attrs, std::vector<int>* in_attrs, std::vector<int>* out_attrs){
	CHECK_EQ(in_attrs->at(0), 0) << "type of gradient should be float";
	CHECK_EQ(in_attrs->at(1), 3) << "type of compressed should be uint8";

    return true;
};

template <typename xpu, typename policy_t>
void TernGradEncode(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs){
    using namespace mxnet_op;
    const TernGradEncode_param& param = nnvm::get<TernGradEncode_param>(attrs.parsed);
    void* stream = get_stream<policy_t>::get(ctx);
    policy_t policy = get_policy<policy_t>::get(stream);
    	const TBlob &gradient_data = inputs[0];
	float* gradient = reinterpret_cast<float*>(gradient_data.dptr<float>());
	uint32 _gradient_size = gradient_data.Size();
	const TBlob &compressed_data = inputs[1];
	uint8* compressed = reinterpret_cast<uint8*>(compressed_data.dptr<uint8>());
	uint32 _compressed_size = compressed_data.Size();

    TernGradEncode_body<policy_t>(
		gradient,
		_gradient_size,
		compressed,
		_compressed_size,
		param.bitwidth,
		policy,
		stream

    );

};

}
}
