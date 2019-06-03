#include "./conv3d_layer_module.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

struct Tensor *cnn_create_conv3d_layer_out(struct Tensor *x, struct Tensor *filter, int stride, int pad){
    long long shape[4] = {x->shapes[0], filter->shapes[0], (x->shapes[2] + 2 * pad - filter->shapes[2]) / stride + 1, (x->shapes[3] + 2 * pad - filter->shapes[3]) / stride + 1};
    return tensor_create_values_deep(shape, 4, 0);
}

void cnn_comput_conv3d_layer_forward(struct Tensor* x, struct Tensor* filter, struct Tensor* bias, int stride, int pad, int padding, struct Tensor* out, int index, int max_index){
    int multipler_out2 = out->shapes[3] * out->shapes[2];
    //multipler_out2 = out_shape[3] * out_shape[2]
    int multipler_out1 = multipler_out2 * out->shapes[1];
    //multipler_out1 = multipler_out2 * out_shape[1]
    int multipler_filter2 = filter->shapes[3] * filter->shapes[2];
    //multipler_filter2 = filter_shape[3] * filter_shape[2]
    int multipler_filter1 = multipler_filter2 * filter->shapes[1];
    //multipler_filter1 = multipler_filter2 * filter_shape[1]
    int multipler_x2 = x->shapes[3] * x->shapes[2];
    //multipler_x2 = x_shape[3] * x_shape[2]
    int multipler_x1 = multipler_x2 * x->shapes[1];
    //multipler_x1 = multipler_x2 * x_shape[1]
    //f1_range = range(filter_shape[1])
    //f2_range = range(filter_shape[2])
    //f3_range = range(filter_shape[3])
    for (int out_index = index * out->size / max_index, out_index_max = (index + 1) * out->size / max_index; out_index < out_index_max; out_index++){
        //for out_index in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        int o0 = out_index / multipler_out1 % out->shapes[0];
        //o0 = out_index // multipler_out1 % out_shape[0]
        int filter_index0 = (out_index / multipler_out2 % out->shapes[1]) * multipler_filter1;
        //filter_index0 = (out_index // multipler_out2 % out_shape[1]) * multipler_filter1
        int x2_tmp = (out_index / out->shapes[3] % out->shapes[2]) * stride - pad;
        //x2_tmp = (out_index // out_shape[3] % out_shape[2]) * stride - pad
        int x3_tmp = (out_index % out->shapes[3]) * stride - pad;
        //x3_tmp = (out_index % out_shape[3]) * stride - pad
        float tmp = 0;
        for(int f1 = 0, f1_max = filter->shapes[1]; f1 < f1_max; f1++){
            //for f1 in f1_range:
            int filter_index1 = filter_index0 + f1 * multipler_filter2;
            //filter_index1 = filter_index0 + f1 * multipler_filter2
            int x_index1 = o0 * multipler_x1 + f1 * multipler_x2;
            //x_index1 = o0 * multipler_x1 + f1 * multipler_x2
            for(int f2 = 0, f2_max = filter->shapes[2]; f2 < f2_max; f2++){
                //for f2 in f2_range:
                int filter_index2 = filter_index1 + f2 * filter->shapes[3];
                //filter_index2 = filter_index1 + f2 * filter_shape[3]
                int x2 = f2 + x2_tmp;
                //x2 = f2 + x2_tmp
                int x_index2 = x_index1 + x2 * x->shapes[3];
                //x_index2 = x_index1 + x2 * x_shape[3]
                if((-1 < x2) && (x2 < x->shapes[2])){
                    //if((-1 < x2) & (x2 < x_shape[2])):
                    for(int f3 = 0, f3_max = filter->shapes[3]; f3 < f3_max; f3++){
                        //for f3 in f3_range:
                        int x3 = f3 + x3_tmp;
                        //x3 = f3 + x3_tmp
                        if((-1 < x3) && (x3 < x->shapes[3]))
                            tmp += x->scalas[x_index2 + x3] * filter->scalas[filter_index2 + f3];
                        else
                            tmp += padding;
                    }
                }
                else{
                    tmp += padding * filter->shapes[3];
                }
            }
        }
        out->scalas[out_index] = tmp + bias->scalas[out_index / multipler_out2 % out->shapes[1]];
    }
}


void cnn_comput_conv3d_layer_backward(struct Tensor* dout, struct Tensor* filter, int stride, int pad, struct Tensor* dx, int index, int max_index){
    int multipler_dout2 = dout->shapes[3] * dout->shapes[2];
    //multipler_dout2 = dout_shape[3] * dout_shape[2]
    int multipler_dout1 = multipler_dout2 * dout->shapes[1];
    //multipler_dout1 = multipler_dout2 * dout_shape[1]
    int multipler_filter2 = filter->shapes[3] * filter->shapes[2];
    //multipler_filter2 = filter_shape[3] * filter_shape[2]
    int multipler_filter1 = multipler_filter2 * filter->shapes[1];
    //multipler_filter1 = multipler_filter2 * filter_shape[1]
    int multipler_dx2 = dx->shapes[3] * dx->shapes[2];
    //multipler_dx2 = dx_shape[3] * dx_shape[2]
    int multipler_dx1 = multipler_dx2 * dx->shapes[1];
    //multipler_dx1 = multipler_dx2 * dx_shape[1]
    
    //dout1_range = range(dout_shape[1])
    for(int dx_index = index * dx->size / max_index, dx_index_max = (index + 1) * dx->size / max_index; dx_index < dx_index_max; dx_index ++){
        //for dx_index in range(index * len(dx_array) // max_index, (index + 1) * len(dx_array) // max_index):
        int dx3 = dx_index % dx->shapes[3];
        //dx3 = dx_index % dx_shape[3]
        int dx2 = dx_index / dx->shapes[3] % dx->shapes[2];
        //dx2 = dx_index // dx_shape[3] % dx_shape[2]
        float tmp = 0; // 이녀석 위치 연구가 필요함.

        int dx3pad = dx3 + pad;
        //dx3pad = dx3 + pad
        int dx2pad = dx2 + pad;
        //dx2pad = dx2 + pad

        int dout_index0 = dx_index / multipler_dx1 * multipler_dout1;
        //dout_index0 = (dx_index // multipler_dx1) * multipler_dout1
        int filter_index1 = dx_index / multipler_dx2 % dx->shapes[1] * multipler_filter2;
        //filter_index1 = (dx_index // multipler_dx2 % dx_shape[1]) * multipler_filter2

        
        //dout3_range = range(max((dx3pad - filter_shape[3]) // stride, -1) + 1, min(-(-(dx3pad + 1)// stride),dout_shape[3]))
        //dout2_range = range(max((dx2pad - filter_shape[2]) // stride, -1) + 1, min(-(-(dx2pad + 1)// stride),dout_shape[2]))
        
        for(int dout1 = 0, dout1_max = dout->shapes[1]; dout1 < dout1_max; dout1++){
            //for dout1 in dout1_range:
            int dout_index1 = dout_index0 + dout1 * multipler_dout2;
            //dout_index1 = dout_index0 + dout1 * multipler_dout2
            int filter_index0 = filter_index1 + dout1 * multipler_filter1;
            //filter_index0 = filter_index1 + dout1 * multipler_filter1

            for(int dout2 = MAX((dx2pad - filter->shapes[2]) / stride, -1) + 1, dout2_max = MIN(-(-(dx2pad + 1) / stride), dout->shapes[2]); dout2 < dout2_max; dout2++){
                //for dout2 in dout2_range:
                int dout_index2 = dout_index1 + dout2 * dout->shapes[3];
                //dout_index2 = dout_index1 + dout2 * dout_shape[3]
                int filter_index2 = filter_index0 + (dx2pad - dout2 * stride) * filter->shapes[3];
                //filter_index2 = filter_index0 + (dx2pad - dout2 * stride) * filter_shape[3]
                for(int dout3 = MAX((dx3pad - filter->shapes[3]) / stride, -1) + 1, dout3_max = MIN(-(-(dx3pad + 1) / stride), dout->shapes[3]); dout3 < dout3_max; dout3++){
                    //for dout3 in dout3_range:
                    tmp += filter->scalas[filter_index2 + dx3pad - dout3 * stride] * dout->scalas[dout_index2 + dout3];
                }
            }
        }
        dx->scalas[dx_index] = tmp;
    }
}

void cnn_comput_conv3d_layer_dfilter(struct Tensor* dout, struct Tensor* x, int stride, int pad, int padding, struct Tensor* dfilter, int index, int max_index){
    //zero = type(dfilter_array[0])(0)
    int mulitpler_dout2 = dout->shapes[3] * dout->shapes[2];
    //multipler_dout2 = dout_shape[3] * dout_shape[2]
    int multipler_dout1 = mulitpler_dout2 * dout->shapes[1];
    //multipler_dout1 = multipler_dout2 * dout_shape[1]
    int multipler_dfilter2 = dfilter->shapes[3] * dfilter->shapes[2];
    //multipler_dfilter2 = dfilter_shape[3] * dfilter_shape[2]
    int multipler_dfilter1 = multipler_dfilter2 * dfilter->shapes[1];
    //multipler_dfilter1 = multipler_dfilter2 * dfilter_shape[1]
    int multipler_x2 = x->shapes[3] * x->shapes[2];
    //multipler_x2 = x_shape[3] * x_shape[2]
    int multipler_x1 = multipler_x2 * x->shapes[1];
    //multipler_x1 = multipler_x2 * x_shape[1]
    
    //dout0_range = range(dout_shape[0])
    //dout2_range = range(dout_shape[2])
    //dout3_range = range(dout_shape[3])

    for(int dfilter_index = index * dfilter->size / max_index, dfilter_index_max = (index + 1) * dfilter->size / max_index; dfilter_index < dfilter_index_max; dfilter_index++){
        //for dfilter_index in range(index * len(dfilter_array) // max_index, (index + 1) * len(dfilter_array) // max_index):
        int x_tmp2 = dfilter_index / dfilter->shapes[3] % dfilter->shapes[2] - pad;
        //x_tmp2 = (dfilter_index // dfilter_shape[3] % dfilter_shape[2]) - pad
        int x_tmp3 = dfilter_index % dfilter->shapes[3] - pad;
        //x_tmp3 = (dfilter_index % dfilter_shape[3]) - pad
        int dout_index1 = dfilter_index / multipler_dfilter1 * mulitpler_dout2;
        //dout_index1 = (dfilter_index // multipler_dfilter1) * multipler_dout2
        int x_index1 = dfilter_index / multipler_dfilter2 % dfilter->shapes[1] * multipler_x2;
        //x_index1 = (dfilter_index // multipler_dfilter2 % dfilter_shape[1]) * multipler_x2
        float tmp = 0;
        //tmp = zero

        for(int dout0 = 0, dout0_max = dout->shapes[0]; dout0 < dout0_max; dout0++){
        //for dout0 in dout0_range:
            int dout_index0 = dout_index1 + dout0 * multipler_dout1;
            //dout_index0 = dout_index1 + dout0 * multipler_dout1
            int x_index0 = x_index1 + dout0 * multipler_x1;
            //x_index0 = x_index1 + dout0 * multipler_x1
            for(int dout2 = 0, dout2_max = dout->shapes[2]; dout2 < dout2_max; dout2++){
                //for dout2 in dout2_range:
                int dout_index2 = dout_index0 + dout2 * dout->shapes[3];
                //dout_index2 = dout_index0 + dout2 * dout_shape[3]
                int x2 = x_tmp2 + dout2 * stride;
                //x2 = x_tmp2 + dout2 * stride
                int x_index2 = x_index0 + x2 * x->shapes[3];
                //x_index2 = x_index0 + x2 * x_shape[3]
                if((-1 < x2) && (x2 < x->shapes[2])){
                    //if((-1 < x2) & (x2 < x_shape[2])):
                    for(int dout3 = 0, dout3_max = dout->shapes[3]; dout3 < dout3_max; dout3++){
                        //for dout3 in dout3_range:
                        int x3 = x_tmp3 + dout3 * stride;
                        //x3 = x_tmp3 + dout3 * stride
                        if((-1 < x3) && (x3 < x->shapes[3])){
                            //if((-1 < x3) & (x3 < x_shape[3])):
                            tmp += x->scalas[x_index2 + x3] * dout->scalas[dout_index2 + dout3];
                            //tmp += x_array[x_index2 + x3] * dout_array[dout_index2 + dout3]
                        }
                        else{
                            //else:
                            tmp += padding * dout->scalas[dout_index2 + dout3];
                            //tmp += padding * dout_array[dout_index2 + dout3]
                        }
                    }
                }
                else{
                    //else:
                    for(int dout3 = 0, dout3_max = dout->shapes[3]; dout3 < dout3_max; dout3++){
                        //for dout3 in dout3_range:
                        tmp += padding * dout->scalas[dout_index2 + dout3];
                        //tmp += padding * dout_array[dout_index2 + dout3]
                    }
                }
            }
        }
        dfilter->scalas[dfilter_index] = tmp;
        //dfilter_array[dfilter_index] = tmp
    }
}
void cnn_comput_conv3d_layer_dbias(struct Tensor* dout, struct Tensor* dbias, int index, int max_index){
    //zero = type(dbias_array[0])(0)

    //dout0_range = range(dout_shape[0])
    //dout23_range = range(dout_shape[2] * dout_shape[3])
    int multipler_dout2 = dout->shapes[3] * dout->shapes[2];
    //multipler_dout2 = dout_shape[3] * dout_shape[2]
    int multipler_dout1 = multipler_dout2 * dout->shapes[1];
    //multipler_dout1 = multipler_dout2 * dout_shape[1]
    for(int dbias_index = index * dbias->size / max_index, dbias_index_max = (index + 1) * dbias->size / max_index; dbias_index < dbias_index_max; dbias_index++){
        //for dbias_index in range(index * len(dbias_array) // max_index, (index + 1) * len(dbias_array) // max_index):
        float tmp = 0;
        //tmp = zero
        int dout_index2 = dbias_index * multipler_dout2;
        //dout_index2 = dbias_index * multipler_dout2
        for(int dout0 = 0, dout0_max = dout->shapes[0]; dout0 < dout0_max; dout0++){
            //for dout0 in dout0_range:
            int dout_index0 = dout_index2 + dout0 * multipler_dout1;
            //dout_index0 = dout_index2 + dout0 * multipler_dout1
            for(int dout23 = 0, dout23_max = dout->shapes[2] * dout->shapes[3]; dout23 < dout23_max; dout23++){
                //for dout23 in dout23_range:
                tmp += dout->scalas[dout_index0 + dout23];
                //tmp += dout_array[dout_index0 + dout23]
            }
        }
        dbias->scalas[dbias_index] = tmp;
        //dbias_array[dbias_index] = tmp
    }
}