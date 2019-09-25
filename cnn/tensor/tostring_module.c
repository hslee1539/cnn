#define _CRT_SECURE_NO_WARNINGS    // sprintf 보안 경고로 인한 컴파일 에러 방지

#include <malloc.h>
#include <stdio.h>
#include "tostring_module.h"

int _tostring_process1(char * out, long long *pos, int *out_index, int out_size, int dim);
int _tostring_process2(char *out, float *array, long long *shape, long long *pos, int *out_index, int out_size, int dim);
void _tostring_process2_stack(long long *shape, long long *pos, int i, int dim);
int _tostring_process3(char * out, long long *pos, int *out_index, int out_size, int dim);
int _tostring_process4(char *out, long long *pos, int *out_index, int out_size, int dim);
int _number2string(float number, char *out, int *out_index, int out_size);


char * tensor_tostring(struct Tensor* tensor, char *out, int out_size){
    long long *pos = calloc(tensor->dim, sizeof(long long));
    int out_index = 0;
    do{
        if(_tostring_process1(out, pos, &out_index, out_size, tensor->dim))
            break; // 더이상 하면 오버플로우 되는 경우
        if(_tostring_process2(out, tensor->scalas, tensor->shapes, pos, &out_index, out_size, tensor->dim))
            break;// 더이상 하면 오버플로우 되는 경우
        if(_tostring_process3(out, pos, &out_index, out_size, tensor->dim))
            break;// 더이상 하면 오버플로우 되는 경우
    }while(_tostring_process4(out, pos, &out_index, out_size, tensor->dim));
    free(pos);
    return out;
}


int _tostring_process1(char * out, long long *pos, int *out_index, int out_size, int dim){
    int count = 0;
    for (int i = 0; i < dim; i++){
        if(pos[i] > 0)
            count = 0;
        else
            count ++;
    }
    if (out_index[0] + dim < out_size - 2){
        for(int i = 0; i < dim - count; i++)
            out[out_index[0]++] = ' ';
        for(int i = 0; i < count; i++)
            out[out_index[0]++] = '[';
        return 0;
    }
    return 1;
}

int _number2string(float number, char *out, int *out_index, int out_size){
    char tmp[] = {0,0,0,0,0,0,0,0,0,0,0};
    sprintf(tmp, "%f", number);
    int overflow = 0;
    int len = 0;
    for(; len < 11; len++){
        if(tmp[len] == 0){
            overflow = 1;
            break;
        }
    }
    if(overflow && (out_size > len + out_index[0])){
        for(int i = 0; i < len; i++)
            out[out_index[0]++] = tmp[i];
        return 0;
    }
    printf("overflow %d\n", out_index[0]);
    return 1;
}

int _tostring_process2(char * out, float * array, long long *shape, long long *pos, int * out_index, int out_size, int dim){
    int point = 0;
    int multipler = 1;
    int i = dim - 1;
    while(i > -1){
        point += multipler * (pos[i] % shape[i]);
        multipler *= shape[i];
        i --;
    }
    if(_number2string(array[point],out, out_index, out_size))
        return 1;
    if(dim > 0){
        for(i = 1; i < shape[dim - 1]; i++){
            if(out_index[0] > out_size - 3)
                return 1;
            out[out_index[0]++] = ',';
            out[out_index[0]++] = ' ';
            if(_number2string(array[point + i],out, out_index, out_size))
                return 1;
        }
        pos[dim - 1] = shape[dim -1];
        _tostring_process2_stack(shape, pos, 1, dim);
        return 0;
    }
    return 1;
}

void _tostring_process2_stack(long long *shape, long long *pos, int i, int dim){
    pos[dim - i] = 0;
    if(dim > i){
        pos[dim -i -1]++;
        if(pos[dim -i -1] == shape[dim -i -1])
            _tostring_process2_stack(shape, pos, i + 1, dim);
    }
}

int _tostring_process3(char *out, long long *pos, int *out_index, int out_size, int dim){
    int count = 0;
    for(int i = 0; i < dim; i++){
        if(pos[i] > 0)
            count = 0;
        else
            count++;
    }
    if(out_index[0] + count < out_size - 3){
        for(int i = 0; i < count; i++)
            out[out_index[0]++] = ']';
        return 0;
    }
    return 1;
}

int _tostring_process4(char *out, long long *pos, int *out_index, int out_size, int dim){
    int count = 0;
    if(out_index[0] + 3 > out_size)
        return 0;
    out[out_index[0]++] = '\n';
    for(int i = 0; i < dim; i++)
        if(pos[i] == 0)
            count++;
    if(dim == count)
        return 0;
    return 1;
}