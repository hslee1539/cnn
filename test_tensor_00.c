#include <stdio.h>

#include "./tensor/main_module.h"

char str[] = "                                                                                                                                                                                                                                                                                                                                   ";
char* tostring(struct Tensor* tensor){
    return tensor_tostring(tensor, str, sizeof(str));
}

void main (void){
    printf("%d process, tensor_create\n", a);
    struct Tensor* test00 = tensor_create();
    printf("test00: %s\n", tostring(test00));
    int shapes3[] = {2,3,4};
    struct Tensor* test01 = tensor_create_gauss_deep(shapes3, 3, 2);
    printf("test01c create!\n");
    printf("test01 : %s\n", tostring(test01));
    struct Tensor* test01c = tensor_create_struct_deep(test01);
    printf("test01c: %s\n", tostring(test01c));
    tensor_shuffle(test01c, 3);
    printf("test01c shuffle: %s\n", tostring(test01c));
    tensor_set(test01c, test01);
    printf("test01c shuffle: %s\n", tostring(test01c));
    tensor_gauss(test01, 4);
    printf("test01c shuffle: %s\n", tostring(test01c));
    tensor_release_deep(test00);
    tensor_release_deep(test01);
    tensor_release_deep(test01c);
}