#include <stdio.h>
#include "THGeneral.h"

int main() {
    printf("Hello, World!\n");
    M_PI;
    USE_BLAS;
    USE_LAPACK;
    printf("start\n");
// $ Error: wrong at /Users/cgl/Desktop/MyProject/learnTorch/torch/lib/TH/test.c:11
//    THError("%s","wrong");
// $ Invalid argument 5: message at /Users/cgl/Desktop/MyProject/learnTorch/torch/lib/TH/test.c:12
//    THArgCheck(0,5,"%s","message");
//$ Error: Assertion `5==3' failed.  at /Users/cgl/Desktop/MyProject/learnTorch/torch/lib/TH/test.c:14
//    THAssert(5==3);
    printf("end\n");
    return 0;
}
