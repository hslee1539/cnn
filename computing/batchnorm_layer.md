[cnn](./../README.md).[computing](./README.md).batchnorm
=======


설명 수정이 필요한 문서입니다. 

# index

# 1.요약
신경망에서 정규화를 하는 레이어 입니다.
정규화는 0차원을 데이터를 가지고 나머지 전체 차원에 대한 정규화를 합니다.

# 2.구조

## 2.1.cnn_comput_batchnorm_layer_forward
정규화의 순전파 연산을 해당 변수를 가지고 수행합니다.

## 2.2.cnn_comput_batchnorm_layer_backward
정규화의 역전파 연산을 해당 변수를 가지고 수행합니다.

## 2.3.index와 max_index
신경망 계산은 양이 많기 때문에 계산을 구역을 나누어 분할을 제어하는 인수입니다. 

### 2.3.1.max_index
정규화에서 계산 구간을 몇 등분 할 것인지 나타내는 정수값 입니다.

### 2.3.2.index
정규화에서 계산 구간에서 몇 번째 구역을 계산할 것인지 나타내는 정수값 입니다.


# 3.작동 예
1. 먼저 필요한 텐서들(dispersion, out, delta out)을 할당을 합니다
2. cnn_comput_batchnorm_layer_forward로 순전파 결과와 분산값을 계산합니다.
3. cnn_comput_batchnorm_layer_backward로 역전파 결과를 계산합니다.

## 3.1.dispersion
정규화를 하기 위해서 입력에 대한 몯ㄴ