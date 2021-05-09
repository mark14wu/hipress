
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

struct floatToUint{
	uint8 bitwidth;
	float* gradient;
	float min;
	float gap;
	int _maximum_index;
	uint32 _t;

floatToUint(
	uint8 bitwidth_,
	float* gradient_,
	float min_,
	float gap_,
	int _maximum_index_,
	uint32 _t_
){
	bitwidth = bitwidth_;
	gradient = gradient_;
	min = min_;
	gap = gap_;
	_maximum_index = _maximum_index_;
	_t = _t_;

}
__host__ __device__
uint8 operator()(int index){
	Random<float> random_float(_t);
	float r;
	uint8 _q=0;
for (int _i = 0; _i < 8/bitwidth; _i++){
    index++;
    if (index < _maximum_index){
r=(gradient[index]-min)/gap+random_float(0,1);
_q = (_q << (bitwidth)) + floor(r);
   } //if
} // for
return _q;

}
};
struct u_smaller{

u_smaller(

){

}
__host__ __device__
float operator()(float a,float b){

if (a<b){
            return a;
}else{
            return b;
};;

}
};
struct u_greater{

u_greater(

){

}
__host__ __device__
float operator()(float a,float b){

if (a>b){
            return a;
}else{
            return b;
};;

}
};
template <typename policy_t>
void TernGradEncode_body(
	float* gradient,
	int32 _gradient_size,
	uint8* compressed,
	int32 _compressed_size,
	uint8 bitwidth,
	policy_t policy,
	void* stream

){
float max;
float min;
float gap;
uint8 tail;
uint8* Q=compressed+(10);
max = thrust::reduce(policy, gradient, gradient+_gradient_size, -99999, u_greater());
min = thrust::reduce(policy, gradient, gradient+_gradient_size, 99999, u_smaller());
gap=(max-min)/((1<<bitwidth)-1);
tail=_gradient_size%(1<<bitwidth);
thrust::transform(policy,thrust::counting_iterator<int32_t>(0),thrust::counting_iterator<int32_t>(0) + (floor(bitwidth*(_gradient_size - 1 + 8/bitwidth)/8)),Q,floatToUint(bitwidth,gradient,min,gap,_gradient_size,std::chrono::high_resolution_clock::now().time_since_epoch().count()));;
get_policy<policy_t>::memcpyIn(compressed+(0), &bitwidth, 1, stream);
get_policy<policy_t>::memcpyIn(compressed+(1), &tail, 1, stream);
get_policy<policy_t>::memcpyIn(compressed+(2), &min, 4, stream);
get_policy<policy_t>::memcpyIn(compressed+(6), &max, 4, stream);
;

};