#include <malloc.h>
#include "./optimizer_module.h"

struct cnn_Optimizer* cnn_create_optimizer(float learning_rate, cnn_optimizer_fpUpdate update){
    struct cnn_Optimizer* newOptimizer = malloc(sizeof(struct cnn_Optimizer));
    newOptimizer->update = update;
    newOptimizer->learning_rate = learning_rate;
    return newOptimizer;
}
void cnn_release_optimizer(struct cnn_Optimizer *optimizer){
    free(optimizer);
}
void cnn_optimizer_update(struct cnn_Optimizer *optimizer, struct cnn_UpdateList *list, int index, int max_index){
    optimizer->update(optimizer, list, index, max_index);
}