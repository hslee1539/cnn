#include "./ada_module.h"
#include <math.h>

void cnn_ada_callback(struct cnn_Optimizer *optimizer, struct cnn_UpdateList *list, int index, int max_index){
    for(int set_index = 0; set_index < list->setSize; set_index++){
        if(list->sets[set_index]->momnt == 0){
            list->sets[set_index]->momnt = tensor_create_values_deep(list->sets[set_index]->value->shapes, list->sets[set_index]->value->dim,0);
        }
        for(int scala_index = index * list->sets[set_index]->value->size / max_index, scala_index_max = (index + 1) * list->sets[set_index]->value->size / max_index; scala_index < scala_index_max; scala_index++){
            list->sets[set_index]->momnt->scalas[scala_index] += list->sets[set_index]->delta->scalas[scala_index] * list->sets[set_index]->delta->scalas[scala_index];
            list->sets[set_index]->value->scalas[scala_index] -= optimizer->learning_rate * list->sets[set_index]->delta->scalas[scala_index] / (sqrtf(list->sets[set_index]->momnt->scalas[scala_index]) + 1e-7f);
        }
    }
}

struct cnn_Optimizer *cnn_create_Ada(float learning_rate){
    return cnn_create_optimizer(learning_rate, cnn_ada_callback);
}
