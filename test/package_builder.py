import sys
import os
import time

dirName = os.path.dirname(__file__)

print("cmd에서 실행해야 컴파일 에러 메세지를 볼 수 있습니다.")
print("패키지를 컴파일 합니다...")
#g++ -shared -lstdc++ -o Sum.dll Sum.cpp -Wl,--output-def,Sum.def,--out-implib,libSum.a
#출처: https://acpi.tistory.com/3 [Test Code]
print(os.popen('gcc -shared -o {0}/lib.dll {0}/lib.c -W -Wall -O2 -Wl,--out-implib,{0}/lib.a'.format(dirName)))
print(os.popen('gcc -o {0}/test.exe -g {0}/test.c -L{0} -llib'.format(dirName)))
#print(os.popen('gcc -O2 -c -W -Wall -pthread {0}/cnn_module.c -o {0}/cnn_module.o'.format(dirName)).read())
#print("컴파일 완료")
#print("동적 라이브러리 파일을 빌드중입니다...")
#print(os.popen('gcc -shared -O2 -pthread {0}/cnn_module.o -o {0}/cnn_module.so'.format(dirName)).read())
#print("빌드 완료")
#print("아무 키나 누르면 종료합니다.")
input()
