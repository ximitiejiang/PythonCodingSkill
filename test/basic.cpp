#include <iostream>

using namespace std;

int main()
{
    /* Q. 如何声明变量？
    一定要指明类型，包括的变量类型有：
    int(4), short int(2), long int(8), unsigned int, unsigned short int, unsigned long int
    float(4), double(8)
    char(1)
    bool(1)
    */ 
    short int a = 1;
    int b = 1;
    long int c =1;

    float t1 = 3.14;
    bool is_ok = false;
    
    char ch = 'z';

    /* Q. 如何检查变量字节长度？
    1. 用sizeof()可以检查字节长度, int/float一样为4，short/char/bool一样为2
    2. 用a.size()可以检查数组长度
    */
    cout << "short int, int, long int size are: "<< sizeof(a) << sizeof(b)<< sizeof(c)<<endl;
    cout << "float size is: " << sizeof(t1) << endl;
    cout << "bool size is:" << sizeof(is_ok) <<endl;
    cout << "char size is:" << sizeof(ch) << endl;

    /* Q. 如何定义自动推断变量类型？
    采用auto类型变量：
    1. 必须要在c++11才支持， 也就是在g++编译时，要增加参数 -std=c++11
    */
    auto kk = 25.0/7;

    /* Q. 如何自定义变量的别名？
    用typedef来自定义一个变量类型的别名
    */
    typedef double FAST;
    FAST m2 = 9876543;

    /* Q. 如何定义常量，与变量有什么区别？
    1. 声明为常量以后，就不能再被程序修改，否则会编译报错，是保护某些数据不被修改的强大方式。
    2. 可用const声明常量
    3. 可用constexpr声明常量表达式: constexpr double get_pi(){ return 22.0/7;}
       但constexpr属于c++11的语法，似乎在我现在vscode里边不被支持
    */
    const double pi = 22.0/7;        // 把变量转变为常量
    
    /* Q. 如何声明枚举类型，有什么用？
    1. 可用于对一组不同变量名进行集合，并可简单的自动顺序赋值。(似乎缺失没有类似python dict的那种对多个变量管理的数据类型)
    2. 调用枚举类型就是直接调用里边的变量名， 而枚举变量名什么作用？
    */
    enum colors
    {
        red=10,
        green=20,
        blue=30,
    };
    cout << green << endl;

    return 0;
}
