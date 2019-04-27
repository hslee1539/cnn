// 모든 c파일을 include 합니다.

// 외부 패키지
#include "../tensor/main_module.c"

    // 외부 패키지만 사용하는 모듈
    //#include "./computing/batchnorm_layer_module.c"
    #include "./computing/fully_connected_layer_module.c"
    #include "./computing/relu_layer_module.c"
    #include "./computing/softmax_layer_module.c"

    #include "./struct/extradata_module.c"  
    #include "./struct/updateset_module.c"

        // 위 패키지를 사용하는 모듈
        #include "./struct/updatelist_module.h"
        #include "./struct/optimizer_module.c"

            // 위 패키지를 사용하는 모듈
            #include "./struct/layer_module.c"
            
                //위 패키지를 사용하는 모듈
                #include "./struct/layer/function/network_layer_module.c"
                #include "./struct/layer/function/fully_connected_layer_module.c"
                #include "./struct/layer/function/activation_function_layer_module.c"

                    #include "./struct/layer/function/relu_layer_module.c"
                    #include "./struct/layer/function/softmax_layer_module.c"

                        #include "./struct/layer/dataset_layer_module.c"    
                        #include "./struct/layer/fully_connected_layer_module.c"
                        #include "./struct/layer/relu_layer_module.c"
                        #include "./struct/layer/softmax_layer_module.c"
                        #include "./struct/layer/network_layer_module.c"


                
                
            

