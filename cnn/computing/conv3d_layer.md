[cnn](./../README.md).[computing](./README.md).conv3d
=======
# index

# 1.요약
컨볼루션 연산을 하는 레이어입니다.

4차원 텐서가 필요하고, 각 차원은 [데이터, 채널, X or Y, X or Y]로 계산합니다.

>주의!! h파일에는 없는 메크로함수 MIN, MAX가 c파일에만 선언이 되어 있습니다. 외부에 노출되는 것을 최소화 하기 위해 이렇게 코딩하였습니다.

# 2.구조

## 2.1.cnn_create_conv3d_layer_out
conv3d 순전파 연산 결과를 저장할 텐서를 만듭니다.

## 2.2.cnn_comput_conv3d_layer_forward
conv3d 순전파 연산을 해당 인수를 가지고 수행합니다.

## 2.3.cnn_comput_conv3d_layer_backward
conv3d 역전파 연산을 해당 인수를 가지고 수행합니다. conv3d의 입력에 대한 미분을 계산합니다.

## 2.4.cnn_comput_conv3d_layer_dfilter
conv3d에서 filter에 대한 미분을 계산합니다.

## 2.5.cnn_comput_conv3d_layer_dbias
conv3d에서 bias에 대한 미분을 계산합니다.

## 2.6.index와 max_index
신경망 계산은 양이 많기 때문에 계산을 구역을 나누어 분할을 제어하는 인수입니다. 

### 2.6.1.max_index
conv3d에서 계산 구간을 몇 등분 할 것인지 나타내는 정수값 입니다.

### 2.6.2.index
conv3d에서 계산 구간에서 몇 번째 구역을 계산할 것인지 나타내는 정수값 입니다.

# 3.작동 예

1. 먼저 필요한 struct Tensor 들(filter, bias, out, delta out, delta filter, delta bias))을 할당합니다.
>참고!! 이중, 순전파 결과를 저장할 struct Tensor 객체는 cnn_create_conv3d_layer_out함수를 통해 만들 수 있습니다.
2. cnn_comput_conv3d_layer_forward 함수로 순전파 결과를 계산합니다.
3. cnn_comput_conv3d_layer_backward 함수로 역전파 결과를 계산합니다.

만약 가중치 업데이트가 필요하면,

4. cnn_comput_conv3d_layer_dfilter 함수로 filter의 미분 계산을 합니다.
5. cnn_comput_conv3d_layer_dbias 함수로 bias의 미분 계산을 합니다.
6. optimizer로 두개의 미분값과 learnig rate로 업데이트 합니다.