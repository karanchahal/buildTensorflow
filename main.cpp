#include "floatTensor.h" 

int main() {

    FloatTensor one(2); // heap declaration
    FloatTensor two(4);
    
    FloatTensor three = one*(&two);

    FloatTensor four(10);

    FloatTensor five = three + (&four);
    five.backward(1);

    cout<<four.grad<<endl; // 1
    cout<<three.grad<<endl; // 1
    cout<<two.grad<<endl; // 1*2 = 2
    cout<<one.grad<<endl; // 1*4 = 4
}