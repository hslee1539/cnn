#include <malloc.h>
#include <string.h>
#include "./tensor_module.h"

struct Tensor* tensor_create_struct_deep(struct Tensor* tensor){
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->scalas = malloc(sizeof(float) * tensor->size);
    out->shapes = malloc(sizeof(long long) * tensor->dim);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;

    out->dim = tensor->dim;
    out->size = tensor->size;
    memcpy(out->scalas, tensor->scalas, tensor->size * sizeof(float));
    memcpy(out->shapes, tensor->shapes, tensor->dim * sizeof(long long));
    return out;
}
struct Tensor* tensor_create_nonstruct(float *scalas, int size, long long *shapes, int dim){
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->information = tensor_INFROMATION_TENSOR_FROM_OTHER;
    out->scalas = scalas;
    out->shapes = shapes;
    out->dim = dim;
    out->size = size;
    return out;
}

struct Tensor* tensor_create_nonstruct_deep(float *scalas, int size, long long *shapes, int dim){
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->scalas = malloc(sizeof(float) * size);
    out->shapes = malloc(sizeof(long long) * dim);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;

    out->dim = dim;
    out->size = size;
    memcpy(out->scalas, scalas, size * sizeof(float));
    memcpy(out->shapes, shapes, dim * sizeof(long long));
    return out;
}

struct Tensor *tensor_create_nonstruct_deep_shape(float *scalas, int size, long long *shapes, int dim){
    struct Tensor *out = malloc(sizeof(struct Tensor));
    out->scalas = scalas;
    out->shapes = malloc(sizeof(long long) * dim);
    out->information = tensor_INFROMATION_SHAPES_NEED_FREE;
    out->dim = dim;
    out->size = size;
    memcpy(out->shapes, shapes, dim * sizeof(long long));
    return out;
}

struct Tensor* tensor_create_values_deep(long long *shapes, int dim, float value){
    int size = 1;
    for(int d = 0; d < dim; d++){
        size *= shapes[d];
    }
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->shapes = malloc(sizeof(long long) * dim);
    out->scalas = malloc(sizeof(float) * size);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;

    out->size = size;
    out->dim = dim;
    
    memcpy(out->shapes, shapes, dim * sizeof(long long));
    for(int i = 0; i < size; i++){
        out->scalas[i] = value;
    }
    return out;
}

struct Tensor* tensor_create(){
    struct Tensor* newTensor = malloc(sizeof(struct Tensor));
    newTensor->scalas = malloc(sizeof(float));
    newTensor->shapes = malloc(sizeof(long long));
    newTensor->information = tensor_INFROMATION_SCALAS_NEED_FREE | tensor_INFROMATION_SHAPES_NEED_FREE;

    newTensor->scalas[0] = 0;
    newTensor->shapes[0] = 0;
    newTensor->dim = 1;
    newTensor->size = 1;
    return newTensor;
}

struct Tensor* tensor_create_value(long long *shapes, int dim, float value){
    int size = 1;
    for(int d = 0; d < dim; d++){
        size *= shapes[d];
    }
    struct Tensor* out = malloc(sizeof(struct Tensor));
    out->shapes = shapes;
    out->scalas = malloc(sizeof(float) * size);
    out->information = tensor_INFROMATION_SCALAS_NEED_FREE;

    out->size = size;
    out->dim = dim;
    
    for(int i = 0; i < size; i++){
        out->scalas[i] = value;
    }
    return out;
}
/*
void tensor_release(struct Tensor* tensor){
    free(tensor);
}*/
void tensor_release_element(struct Tensor* tensor){
    if(tensor->information & tensor_INFROMATION_SHAPES_NEED_FREE){
        free(tensor->shapes);
        tensor->information ^= tensor_INFROMATION_SHAPES_NEED_FREE;
    }
    if(tensor->information & tensor_INFROMATION_SCALAS_NEED_FREE){
        free(tensor->scalas);
        tensor->information ^= tensor_INFROMATION_SCALAS_NEED_FREE;
    }
    tensor->dim = 0;
    tensor->size = 0;
}
void tensor_release_deep(struct Tensor* tensor){
    if(tensor->information & tensor_INFROMATION_SHAPES_NEED_FREE){
        free(tensor->shapes);
        tensor->information ^= tensor_INFROMATION_SHAPES_NEED_FREE;
    }
    if(tensor->information & tensor_INFROMATION_SCALAS_NEED_FREE){
        free(tensor->scalas);
        tensor->information ^= tensor_INFROMATION_SCALAS_NEED_FREE;
    }
    free(tensor);
}

void tensor_reshape_deep(struct Tensor* tensor, long long *shapes, int dim){
    if(tensor->information & tensor_INFROMATION_SHAPES_NEED_FREE){
        free(tensor->shapes);
        tensor->information ^= tensor_INFROMATION_SHAPES_NEED_FREE;
    }
    tensor->information |= tensor_INFROMATION_SHAPES_NEED_FREE;
    tensor->shapes = malloc(sizeof(long long) * dim);
    memcpy(tensor->shapes, shapes, sizeof(long long) * dim);
}

void tensor_set(struct Tensor* tensor, struct Tensor* source){
    if(tensor->information & tensor_INFROMATION_SHAPES_NEED_FREE){
        free(tensor->shapes);
        tensor->information ^= tensor_INFROMATION_SHAPES_NEED_FREE;
    }
    if(tensor->information & tensor_INFROMATION_SCALAS_NEED_FREE){
        free(tensor->scalas);
        tensor->information ^= tensor_INFROMATION_SCALAS_NEED_FREE;
    }
    tensor->scalas = source->scalas;
    tensor->shapes = source->shapes;
    tensor->dim = source->dim;
    tensor->size = source->size;
}

void tensor_referTo(struct Tensor *tensor, struct Tensor *source){
    if(tensor->information & tensor_INFROMATION_SHAPES_NEED_FREE){
        free(tensor->shapes);
        tensor->information ^= tensor_INFROMATION_SHAPES_NEED_FREE;
    }
    if(tensor->information & tensor_INFROMATION_SCALAS_NEED_FREE){
        free(tensor->scalas);
        tensor->information ^= tensor_INFROMATION_SCALAS_NEED_FREE;
    }
    tensor->scalas = source->scalas;
    tensor->shapes = source->shapes;
    tensor->dim = source->dim;
    tensor->size = source->size;
}