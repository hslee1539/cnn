import sys
import os
import time

main_c = './cnn/main_module.c'

print("cmd에서 실행해야 컴파일 에러 메세지를 볼 수 있습니다.")
print("패키지를 컴파일 합니다...")
print(os.popen('gcc -O2 -c -W -Wall -pthread ' + main_c + ' -o ./cnn_module.o').read())
print("컴파일 완료")
print("동적 라이브러리 파일을 빌드중입니다...")
print(os.popen('gcc -shared -O2 -pthread ./cnn_module.o -o ./cnn_module.so').read())
print("빌드 완료")
print("아무 키나 누르면 종료합니다.")
input()
