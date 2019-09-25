#include "./batchnorm_layer_module.h"
#include <math.h>

void cnn_comput_batchnorm_layer_forward(struct Tensor *x, struct Tensor *dispersion, struct Tensor *out, int index, int max_index){
    long long N = x->shapes[0];
    // 데이터 하나의 크기
    // x_array에는 의미 있는 텐서들이 N개가 있고, N개를 나누면 의미 있는 텐서들의 크기가 됨.
    long long D = x->size / N;

    for(long long data_index = index * D / max_index, data_index_max = (index + 1) * D /max_index; data_index < data_index_max; data_index++){
        //평균 계산
        float mean = 0;
        for(long long i = 0; i < N; i++){
            mean += x->scalas[i * D + data_index];
        }
        mean /= N;

        //분산 계산
        //float dispersion = 0;
        dispersion->scalas[data_index] = 0;
        for(long long i = 0; i < N; i++){
            dispersion->scalas[data_index] += (x->scalas[i * D + data_index] - mean) * (x->scalas[i * D + data_index] - mean);
        }
        //dispersion = dispersion / N

        //x를 평균0 분산1로 변환 계산(out_array에 저장)
        dispersion->scalas[data_index] = sqrtf(dispersion->scalas[data_index] / N + 10e-7f);

        for(long long i = 0; i < N; i++){
            out->scalas[i * D + data_index] = (x->scalas[i * D + data_index] - mean) / dispersion->scalas[data_index];
        }
    }
}

void cnn_comput_batchnorm_layer_backward(struct Tensor *dout, struct Tensor *out, struct Tensor *dispersion, struct Tensor *dx, int index, int max_index){
    long long D = dispersion->size;
    // 따라서 dx.length에 D를 나누면, 의미 있는 텐서들의 데이터 수가 됨.
    long long N = dout->size / D;
    long long data_index = index * D / max_index;
    long long data_index_max = (index + 1) * D / max_index;
    

    for(; data_index < data_index_max; data_index++){
        // dx에서 분산 노드로 가는 미분 값
        float tmp1 = 0;
        for(long long i = 0; i < N; i++){
            tmp1 += out->scalas[i * D + data_index] * dout->scalas[i * D + data_index];
        }
        //평균 노드에서 out으로 가는 미분 값
        float tmp2 = 0;
        for(long long i = 0; i < N; i++){
            tmp2 += tmp1 * out->scalas[i * D + data_index] / N - dout->scalas[i * D + data_index];
        }
        tmp2 /= N;
        
        for(long long i = 0; i < N; i++){
            dx->scalas[i * D + data_index] = (tmp2 - tmp1 * out->scalas[i * D + data_index] / N + dout->scalas[i * D + data_index]) / dispersion->scalas[data_index];
        }
    }
}