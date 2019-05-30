import sys
import os
import time

main_c = './cnn/main_module.c'

print(os.popen('gcc -O2 -c ' + main_c + ' -o ./cnn_module.o').read())
print(os.popen('gcc -shared -O2 ./cnn_module.o -o ./cnn_module.so').read())
time.sleep(5)
