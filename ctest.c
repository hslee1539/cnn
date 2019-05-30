#include <stdio.h>

float print(float * array, int size);

float print(float * array, int size){
    return array[0];
}


#define A(a) a
void main(void){
    int a = 10;
    A(a) += 10;
    printf("%d", A(a));
}