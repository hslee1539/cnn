#pragma once

#define tensor_INFROMATION_TENSOR_FROM_OTHER    0
#define tensor_INFROMATION_SCALAS_NEED_FREE     0b1
#define tensor_INFROMATION_SHAPES_NEED_FREE     0b10

/// 텐서 구조체입니다. 1차원 배열로 관리합니다.
struct Tensor;

struct Tensor{
    // scalas의 배열 수
    int size;
    // shapes의 배열 수 (= 차원과 같은 의미)
    int dim;
    float *scalas;
    long long *shapes;
    int information;
};///ㅂㅈㄷ

/// 상수 취급 하시오.
//struct Tensor tensor_NULL;


/// 해당 객체로 깊은 복사를 하여 객체를 만듭니다.
struct Tensor* tensor_create_struct_deep(struct Tensor* tensor);
/// 해당 객체를 참조 복사를 하여 객체를 만듭니다.

/// 깊이 복사하지 않고 참조하여 객체를 만듭니다.
struct Tensor* tensor_create_nonstruct(float *scalas, int size, long long *shapes, int dim);
/// 객체를 깊이 복사하여 객체를 만듭니다.
struct Tensor* tensor_create_nonstruct_deep(float *scalas, int size, long long *shapes, int dim);
/// 객체를 스칼라는 참조하고, shape은 깊이 복사하여 만듭니다.
struct Tensor *tensor_create_nonstruct_deep_shape(float *scalas, int size, long long *shapes, int dim);
/// 해당 값으로 객체를 만듭니다.
struct Tensor* tensor_create_values_deep(long long *shapes, int dim, float value);
/// 0값을 가지는 스칼라를 만듭니다.
struct Tensor* tensor_create();
///해당 값으로 객체를 만듭니다. shapes의 경우, 참조하고, scalas는 새롭게 할당되어 만들어 집니다.
struct Tensor* tensor_create_value(long long *shapes, int dim, float value);
///외부 참조된 scalas나 shapes을 제외하고 반환을 합니다.(비추천 함수)
///참조가 필요하면 tensor_referTo로 참조를 하세요.
void tensor_release_element(struct Tensor* tensor);
///외부 참조된 scalas나 shapes은 제외하고 이 객체와 내부 변수들을 깊이 반환합니다.
void tensor_release_deep(struct Tensor* tensor);
///tensor::shape이 외부 참조된 것이 아니면 shapes이면 반환하고, shapes을 깊은 복사합니다.
void tensor_reshape_deep(struct Tensor* tensor, long long *shapes, int dim);
///scalas나 shapes이 외부 참조된 것이 아니면 해당 되는 것을 반환하고 source를 참조합니다.(새로운 버전 : tensor_referTo)
void tensor_set(struct Tensor* tensor, struct Tensor* source);
///요약:
///tensor가 source의 내용으로 참조를 합니다.
///자세한 내용:
///참조 전에 tensor가 가지고 있던 정보는 tensor->information을 참고하여 메모리를 반환합니다.
///tensor객체를 tensor_release_deep으로 메모리를 반환하여도 참조된 scalas랑 shapes는 반환하지 않습니다.
///반대로 source가 반환이 되면 참조된 tensor 객체는 source에 의해 반환된 것을 참조하기 때문에 위험해 질 수 있습니다.
void tensor_referTo(struct Tensor *tensor, struct Tensor *source);