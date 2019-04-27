/// 가장 기초가 되는 신경망의 레이어 구조체입니다.
struct cnn_Layer2;

struct cnn_Layer2{
    /// 고유 식별 아이디입니다. 해당되는 매크로 문자열을 가르킵니다.
    char*       id;
    /// 이 레이어의 순전파 결과값입니다.
    float*      out;
    int         out_size;
    int*        out_shape;
    int         out_dim;

    float*      dx;
    int         dx_size;
    int*        dx_shape;
    int         dx_dim;


    
}