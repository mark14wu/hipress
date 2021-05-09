#include <stdio.h>
#include <stdlib.h>
#include "ZQ_CPP_LIB/time_cost.hpp"

int main(void){
    zq_cpp_lib::time_cost zt;
    zt.start();
    unsigned long int size = (1ULL<<27)*16;
    int *p = (int*)malloc(size);
    free(p);
    zt.record("first malloc and free");
    for (int i = 0; i < 1000; i++){
        p = (int*)malloc(size);
        free(p);
    }
    zt.record("1000 times of malloc and free");
    zt.print_by_ms();
    return 0;

}