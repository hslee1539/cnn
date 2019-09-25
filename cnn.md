[cnn](./../README.md)
=========

# index

# 1.요약
tensor 패키지를 제외한 cnn의 전체 패키지입니다. 이 디렉토리에서는 외부에서 import 혹은, include를 할 때 쉽게 하기 위한 모듈들을 가지고 있습니다.

# 2.구조
이 디렉토리와 내부 모든 디렉토리는
모듈화 된 코드 파일들은 끝에 _module을 써서 만들었습니다. -> 폴더와 모듈의 이름을 구별하기 위해

모들 디렉토리에는 import_module.c, import_module.h 모듈을 가지고 있어, 그 디렉토리의 모든 기능을 include하는 식의 구조를 가집니다. -> 컴파일 및 include할때 쉽게 하기 위해서
## 2.1.파일 및 폴더 구조

cnn

    computing       : cnn에서 필요한 수학적인 계산 패키지입니다.
    struct          : cnn의 구조체 패키지입니다.
    util            : cnn의 분류하기가 애매한 부가 기능 패키지입니다.

    __init__.py     : python 문법상 필요한 더미 파일입니다.
    import_module.c : 내부 디렉토리를 포함해서 모든 c 파일을 include 한 모듈입니다.
    import_module.h : 내부 디렉토리를 포함해서 모든 h 파일을 include 한 모듈입니다.

    
