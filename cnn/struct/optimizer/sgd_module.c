#include "./sgd_module.h"

void cnn_sgd_callback(struct cnn_Optimizer *optimizer, struct cnn_UpdateList *list, int index, int max_index){
    for(int set_index = 0; set_index < list->setSize; set_index++){
        for(int scala_index = index * list->sets[set_index]->value->size / max_index, scala_index_max = (index + 1) * list->sets[set_index]->value->size / max_index; scala_index < scala_index_max; scala_index++){
            list->sets[set_index]->value->scalas[scala_index] -= optimizer->learning_rate * list->sets[set_index]->delta->scalas[scala_index];
        }
    }
}

struct cnn_Optimizer *cnn_create_SGD(float learning_rate){
    return cnn_create_optimizer(learning_rate, cnn_sgd_callback);
}
