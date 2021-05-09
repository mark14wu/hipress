
#include <chrono>
#include <stdint.h>
#include <thrust/copy.h> //copy_if
#include <thrust/execution_policy.h> //thrust::device
#include <thrust/functional.h> //greater<float>
#include <thrust/iterator/counting_iterator.h> // counting_iterator
#include <thrust/random.h>
#include <thrust/sort.h> //sort()
#include <thrust/transform.h> //trnasform

#include "naive_random.hpp"
#include "get_policy_general.h"
using namespace zq_cpp_lib::operate_memory;

#define Mod(a,b) ((a)%(b))


#define uint8 uint8_t
#define uint32 uint32_t
#define int32 int32_t

struct uintToFloat{
	uint8* Q;
	uint8 bitwidth;
	float gap;
	float min;

uintToFloat(
	uint8* Q_,
	uint8 bitwidth_,
	float gap_,
	float min_
){
	Q = Q_;
	bitwidth = bitwidth_;
	gap = gap_;
	min = min_;

}
__host__ __device__
float operator()(int index){

return ((Q[static_cast<int>((index)*bitwidth/8)]>>(Mod((index), 8/bitwidth)*bitwidth))&((1<<bitwidth)-1))*gap+min;

}
};
template <typename policy_t>
void TernGradDecode_body(
	uint8* compressed,
	int32 _compressed_size,
	float* gradient,
	int32 _gradient_size,
	policy_t policy,
	void* stream

){
uint8 bitwidth;
uint8 tail;
float min;
float max;
uint8* Q = reinterpret_cast<uint8*>(compressed+10);
float gap;
int true_Q_size;
get_policy<policy_t>::memcpyOut(&bitwidth, compressed+0, 1, stream);
get_policy<policy_t>::memcpyOut(&tail, compressed+1, 1, stream);
get_policy<policy_t>::memcpyOut(&min, compressed+2, 4, stream);
get_policy<policy_t>::memcpyOut(&max, compressed+6, 4, stream);
;
gap=(max-min)/((1<<bitwidth)-1);
true_Q_size=(8*floor(_compressed_size + 7/8) - 80)/bitwidth-tail;
thrust::transform(policy,thrust::counting_iterator<int32_t>(0),thrust::counting_iterator<int32_t>(0) + (true_Q_size),gradient,uintToFloat(Q,bitwidth,gap,min));;

};