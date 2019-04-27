#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "./random_module.h"

float gaussianRandom(void) {
    float v1, v2, s;
    do {
        v1 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        v2 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);
    s = sqrtf( (-2 * logf(s)) / s );
    return v1 * s;
}

struct Tensor* tensor_create_gauss_deep(int *shapes, int dim, int seed){
    int size = 1;
    for(int d = 0; d < dim; d++){
        size *= shapes[d];
    }
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->shapes = malloc(sizeof(int) * dim);
    out->scalas = malloc(sizeof(float) * size);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;
    out->size = size;
    out->dim = dim;
    
    memcpy(out->shapes, shapes, dim * sizeof(int));
    
    srand(seed);
    for(int i = 0; i < size; i++){
        out->scalas[i] = gaussianRandom();
    }
    return out;
}

struct Tensor* tensor_create_random_deep(int *shapes, int dim, int seed, float min, float range){
    int size = 1;
    for(int d = 0; d < dim; d++){
        size *= shapes[d];
    }
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->shapes = malloc(sizeof(int) * dim);
    out->scalas = malloc(sizeof(float) * size);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;

    out->size = size;
    out->dim = dim;
    
    memcpy(out->shapes, shapes, dim * sizeof(int));
    
    srand(seed);
    for(int i = 0; i < size; i++){
        out->scalas[i] = (float) rand() / RAND_MAX * range - min;
    }
    return out;
}
void tensor_gauss(struct Tensor* tensor, int seed){
    srand(seed);
    for(int i = 0; i < tensor->size; i++){
        tensor->scalas[i] = gaussianRandom();
    }
}
void tensor_random(struct Tensor* tensor, int seed, float min, float range){
    srand(seed);
    for(int i = 0; i < tensor->size; i++){
        tensor->scalas[i] = (float) rand() / RAND_MAX * range - min;
    }
}
void tensor_shuffle(struct Tensor* tensor, int seed){
    int random_index;
    float tmp;
    srand(seed);
    for(int scala_index = 0; scala_index < tensor->size; scala_index++){
        random_index = rand() % (tensor->size - scala_index) + scala_index;
        tmp = tensor->scalas[scala_index];
        tensor->scalas[scala_index] = tensor->scalas[random_index];
        tensor->scalas[random_index] = tmp;
    }
}
