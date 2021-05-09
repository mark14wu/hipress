#ifndef NAIVE_RANDOM_HPP_
#define NAIVE_RANDOM_HPP_

#include <stdint.h>
#include <iostream>
using namespace std;


#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


template<typename T>
class Random{
private:
    uint32_t _a;
    uint32_t _c;
    uint32_t _x;
public:
    CUDA_HOSTDEV
    Random(const uint32_t seed){
        _a=1103515245;
        _c=12345;
        _x = seed;
    }
    CUDA_HOSTDEV
    uint32_t rand(){
        _x = _a*_x + _c;
        return _x;
    }

    CUDA_HOSTDEV
    T operator()(T lower_bound, T upper_bound){
        return static_cast<T>(rand()%(upper_bound-lower_bound+1))+lower_bound;
    }
};
template<>
class Random<float>{
private:
    uint32_t _a;
    uint32_t _c;
    uint32_t _x;

    uint32_t _max;
    double _max_double;
public:
    CUDA_HOSTDEV
    Random(const uint32_t seed){
        _a=1103515245;
        _c=12345;
        _x = seed;
        _max = ~0;
        _max_double = static_cast<double>(_max);
    }
    CUDA_HOSTDEV
    uint32_t rand(){
        _x = _a*_x + _c;
        return _x;
    }
public:
    CUDA_HOSTDEV
    float operator()(float lower_bound, float upper_bound){
        return static_cast<float>(rand()/(_max_double/(upper_bound-lower_bound)))+lower_bound;
    }
};

#endif