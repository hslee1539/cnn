#pragma once

typedef struct Tensor{
    // scalas의 배열 수
    int size;
    // shapes의 배열 수 (= 차원과 같은 의미)
    int dim;
    float *scalas;
    int *shapes;
}Tensor;

typedef Tensor *pTensor;

pTensor tensor_create_deepCopy_struct(pTensor tensor);
pTensor tensor_create_deepCopy_nonstruct(int size, int dim, float *scalas, int *shapes);
pTensor tensor_create_values(int size, int dim, int value);
pTensor tensor_create_gauss(int size, int dim, int seed);
pTensor tensor_create_random(int size, int dim, int seed);
void tensor_release_deep(pTensor tensor);