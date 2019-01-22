#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:18:35 2019

@author: ubuntu
"""
"""公式参考百度百科：趋肤深度
以及：https://wenku.baidu.com/view/b1e5b96490c69ec3d5bb75bc.html
"""

from math import sqrt, pi
def relative_df(f, r):
    """ 计算趋肤深度df = 6.61/(sqrt(f)) cm
    Args:
        f(hz)
    Returns:
        df(mm)
    """
    if f == 0:
        f = 0.001
    df = 10*6.61/sqrt(f)
    if df > r:
        print('df is big enough, no affect to cable')
    else:
        print('df will affect cable S')
    return df

def relative_S(r,df):
    """计算等效截面积"""
    sf = pi*(r**2 -(r-df)**2)
    s = pi*r**2
    return (sf, s)

def relative_R(p,l,r,df,f):
    """计算等效电阻
    Args:
        p(ohm.m/mm^2), resistance coef. at 20deg
        l(m), cable length
        r(mm), cable radian 
        df(mm)
        f(hz)
    Returns:
        rf()
    """
    if f == 0:
        f = 0.001
    rf = p*l / (pi*(r**2 - (r - df)**2))
    r0 = p*l/(pi*r**2)
    return (rf, r0)
    
if __name__ == '__main__':
#    f = 200e3  # hz
    f = 85e3    # 85khz
    p = 0.01749 # Ohms.m/mm2
    l = 0.15    # cable lenth
    r = 1.4     # cable radius(r=1.4大概6mm2的导线)
    
    df = relative_df(f,1)
    sf, s = relative_S(r,df)
    
    rf, r0 = relative_R(p,l,r,df,f)
    print('df = {:.6f}mm'.format(df))
    print('S= {:.6f}mm2, Sf = {:.6f}mm2'.format(s, sf))
    print('Res={:.6f}Ohms, Resf = {:.6f}Ohms'.format(r0, rf))
    
    # case 2: 该趋肤深度的公式仅针对整根电缆，对于很细的单丝漆包线线计算出来不对的
    print('--'*20)
    f = 85e3    # 85khz
    p = 0.01749 # Ohms.m/mm2    1.678
    l = 0.15    # cable lenth
    r = 0.025     # cable radim(d=0.05单丝)
    num = 4200    # 总根数
    
    df = relative_df(f,1)
    sf, s = relative_S(r,df)
    
    rf, r0 = relative_R(p,l,r,df,f)
    print('df = {:.6f}mm'.format(df))
    print('S= {:.6f}mm2, Sf = {:.6f}mm2'.format(s, sf))
    print('total_S= {:.6f}mm2, total_Sf = {:.6f}mm2'.format(num*s, num*sf))
    print('Total_Res={:.6f}Ohms, total_Resf = {:.6f}Ohms'.format(num*r0, num*rf))
    
    # case 3: 另一种计算趋肤深度的公式，对很细的线也不行，因为算出来df已经超过单丝直径了
    def new_df(f):  # f 取MHz
        """修正公式： df(um) = sqrt(p/(pi*f0*ur*u0))， 
        但这种算出来的0.05单丝直径的线df都大于r了，相当于没影响，所以也不能用。
        https://www.pasternack.com/t-calculator-skin-depth.aspx
        Args:
            p(uOhms cm, 1.678), fo(Mhz), ur(0.999991), u0(4*pi*e-7)
        Returns:
            df(um)
        """
        df = sqrt(1.678/(100*pi*f*0.999991*4*pi*1e-7))
        return df

    print('--'*20)
    print('new_df = {}um'.format(new_df(0.085)))
    ndf = new_df(0.085)/1000 # um to mm
    
    p = 0.01749 # Ohms.m/mm2    1.678
    l = 0.15    # cable lenth
    r = 0.025     # cable radim(d=0.05单丝)
    num = 4200    # 总根数
    
    sf, s = relative_S(r,ndf)

    rf, r0 = relative_R(p,l,r,ndf,f)
    print('df = {:.6f}mm'.format(ndf))
    print('S= {:.6f}mm2, Sf = {:.6f}mm2'.format(s, sf))
    print('total_S= {:.6f}mm2, total_Sf = {:.6f}mm2'.format(num*s, num*sf))
    print('Total_Res={:.6f}Ohms, total_Resf = {:.6f}Ohms'.format(num*r0, num*rf))
    