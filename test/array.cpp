#include <iostream>
#include <vector>
using namespace std;

int main()
{
    /* Q. 声明数组的多少种方法？
    1. 作为静态数组声明，int aa[] = {}; 类似python的list, 可以一维和高纬，只是把[]换成{}
    2. 作为动态数组声明，vector<float> bb (),这种数组可以无所谓尺寸因为可以在最后加元素
        - 需要include <vector>
        - 用push_back()方法在末尾随意添加元素
    */
    int aa [5] = {1,2,3,4,5}; 
    int bb [5] = {};                 //默认会用0初始化
    int cc [] = {1,2,3};
    int dd [2][4] = {{1,2}, {3,4}};  // 多维数组，没有初始化输入的都用0初始化

    vector<int> ee (3);     // 动态数组
    ee[0] = 5;
    ee.push_back(15);       // 用push_back()方法在最后添加元素，注意：只会在初始化大小之外的最后。
    cout << "ee[0] = " << ee[0] << ", ee[1]=" << ee[1] <<", ee[3]=" << ee[3] << endl;
    cout << "ee size is: "<< ee.size() <<endl;  // e.size()函数是专给vector用的？

    /* Q. 数组的切片如何进行？
    */
    int b = aa[3];
    cout << "aa[3] = " << b << endl;
}