import numpy as np
from numba import jit, float64
import scipy.constants as C
import datetime
import os
#import matplotlib.colors as mcolors
from daf_para import DP
from numpy import zeros,pi,random
from PlotSaveResult import PlotSaveResult
"""
程序介绍：

"""
"""文件系统"""
Time_Now_Str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
Root_File_Name = r"E:\Lion\MDITpy\File"
Root_Floder_New =  Root_File_Name + "\\" + Time_Now_Str[0:10]
File_Time = Time_Now_Str

save_csv = True
special_string = True


if os.path.isdir(Root_Floder_New):
    pass
else:
    os.mkdir(Root_Floder_New)

if special_string:
    Special_String = "03-be"
    Root_Floder_Compared = Root_Floder_New + "\\" + Special_String
    if os.path.isdir(Root_Floder_Compared):
        pass
    else:
        os.mkdir(Root_Floder_Compared)
    Floder_Name = Root_Floder_Compared + "\\" + File_Time
else:
    Floder_Name = Root_Floder_New + "\\" + File_Time

if save_csv:
    os.mkdir(Floder_Name)
    print("---  creat a new folder...  ---")

"""常数"""
# constants
kb = C.k
e = C.e
epsilon0 = C.epsilon_0

Def_Para = DP()

"""物理参数"""
M = Def_Para.Mass_ion
N = Def_Para.Number_ion
Q = Def_Para.Q_ion
T_ini_ion = Def_Para.T_ion
T_background_gas = Def_Para.T_background_gas
M_background_gas = Def_Para.M_background_gas

dt = Def_Para.dt
endtime = Def_Para.end_time
Nion = np.sum(N)
M_ratio = M[0]/M[1]
if N[1] != 0:
    MM = np.vstack(([[M[0]]]*N[0],[[M[1]]]*N[1]))
    M_list = np.hstack((MM,MM,MM))
    Q_list = np.ones(Nion)
    Q_list[0:N[0]] = Q_list[0:N[0]]*Q[0]
    Q_list[N[0]::] = Q_list[N[0]::]*Q[1]
else:
    M_list = np.ones((N[0],3))*M[0]
    Q_list = np.ones(N[0])*Q[0]


# time parameters
init_Step = Def_Para.init_step
init = Def_Para.init_Ture


Nstep = int(round(endtime/dt))
endtime = Nstep * dt
Data_N_index = 2000000
if Data_N_index > Nstep:
    Data_N_index = Nstep
else:
    pass

# laser cooling parameters
alpha = Def_Para.beta_laser #

# random force coefficient
beta = Def_Para.beta_random
speed_incre = Def_Para.speed_incre

# trap parameters
fre = Def_Para.f     #囚禁频率
Urf = Def_Para.Urf   #射频峰峰值
omega = fre*2.0*pi 
r0 = Def_Para.r0    
z0 = Def_Para.z0
gfactor_z = Def_Para.gfactor_z  #轴向结构因子
Uend = Def_Para.Uend         #帽极电压

Average_V_Time = int(round(1/fre/dt))
Average_sec_Time = 5
Average_Time = Average_V_Time*Average_sec_Time


#############
"""选择力"""
FTrap_Pseudo_True = Def_Para.FTrap_Pseudo_True
FCoulomb_True = Def_Para.FCoulomb_True
FLaser_True = Def_Para.FLaser_True
FRandom_True = Def_Para.FRandom_True

"""囚禁参数计算"""
ax_Lc,ax_Sy,ax_Sy,qx_Lc,qx_Sy = Def_Para.aq_calculation()

"""赝势梯度"""
Pes_Cool_x = -1 * Q[0] * ((Q[0] * Urf**2 / (2 * M[0] * omega**2 * r0**4)) - gfactor_z*Uend/z0**2)
Pes_Cool_y = -1 * Q[0] * ((Q[0] * Urf**2 / (2 * M[0] * omega**2 * r0**4)) - gfactor_z*Uend/z0**2)
Pes_Cool_z =  -1 * 2 * Q[0] * gfactor_z * Uend/(z0**2)

Pes_Sy_x = -1 * Q[1] * (Q[1] * Urf**2 / (2 * M[1] * omega**2 * r0**4) - gfactor_z*Uend/z0**2)
Pes_Sy_y = -1 * Q[1] * (Q[1] * Urf**2 / (2 * M[1] * omega**2 * r0**4) - gfactor_z*Uend/(z0**2))
Pes_Sy_z =  -1 * 2 * Q[1] * gfactor_z * Uend/z0**2

"""力模型"""
#@jit(float64[:,:](float64[:,:], float64))
@jit
def Ftrap(r, t):
    """赝势模型"""
    F_trap = zeros((Nion, 3))
    F_trap[0:N[0],:] = np.array([Pes_Cool_x,Pes_Cool_y,Pes_Cool_z])*r[0:N[0],:]
    if N[1] != 0:
        F_trap[N[0]:,:] = np.array([Pes_Sy_x,Pes_Sy_y,Pes_Sy_z])*r[N[0]:,:]
    else:
        pass
    return F_trap

#@jit(float64[:,:](float64[:,:], float64))
@jit
def Ftrap_cos(r, t):
    gfactor_z_g = gfactor_z
    ma_x_Cool = -1 * Q[0] * ((Urf*np.cos(omega*t)/r0**2) - gfactor_z_g*Uend/z0**2)
    ma_y_Cool = -1 * Q[0] * ((-1 * Urf*np.cos(omega*t)/r0**2) - gfactor_z_g*Uend/z0**2)
    ma_z_Cool = -1 * 2 * gfactor_z *  Q[0] * Uend / z0**2
    
    ma_x_SY = -1 * Q[1] * ((Urf*np.cos(omega*t)/r0**2) - gfactor_z_g*Uend/z0**2)
    ma_y_SY = -1 * Q[1] * ((-1 * Urf*np.cos(omega*t)/r0**2) - gfactor_z_g*Uend/z0**2)
    ma_z_SY = -1 * 2 * Q[1] * gfactor_z_g * Uend / z0**2
        
    F_trap = zeros((Nion, 3))
    F_trap[0:N[0],:] = np.array([ma_x_Cool,ma_y_Cool,ma_z_Cool])*r[0:N[0],:]
    if N[1] != 0:
        F_trap[N[0]:,:] = np.array([ma_x_SY,ma_y_SY,ma_z_SY])*r[N[0]:,:]
    else:
        pass
    return F_trap

#@jit(float64[:,:](float64[:,:],float64[:,:],float64))
@jit
def Fcoulomb(r,Q_list_C,Nall):
    """库仑力模型"""
    Ftemp = zeros((Nall, 3))
    kG = 1/(4.0*pi*epsilon0)
    for i in range(Nall):
        for j in range(Nall):
            if i != j:
                Ftemp[i] = Ftemp[i] + kG *Q_list_C[i]*Q_list_C[j] * (r[i]-r[j])/(np.sum((r[i]-r[j])**2)**1.5)
            else:
                pass
    return Ftemp  #单位是N

#@jit(float64[:,:](float64[:,:]))
@jit
def Flaser(v):
    """激光冷却力模型"""
    F = -alpha*v
    if N[1] != 0 and init == False:
        F[N[0]:,:] = zeros((N[1],3))
    return F

#@jit(float64[:,:]())
@jit
def Frandom():
    """杂散力模型"""
    Dirf = random.rand(Nion, 3)-0.5
    F_random = beta*(Dirf/np.abs(Dirf))
    if N[1] != 0:
        F_random[N[0]:,:] = F_random[N[0]:,:] / (M_ratio)
    return F_random

#@jit(float64[:,:]())
@jit
def Random_Guass_Force():
    Random_Guass_Force = random.normal(0.0, np.sqrt(kb*T_background_gas/2/C.u), (Nion, 3))/1e4 * beta
    if N[1] != 0:
        Random_Guass_Force[N[0]::] = Random_Guass_Force[N[0]::] * M_ratio
    return Random_Guass_Force * M_list / dt

"""积分算法"""
@jit
def cal_acc(r,v,t):
    F = np.add(zeros((Nion,3)),Ftrap(r,t))
    if FCoulomb_True:
        F = np.add(F,Fcoulomb(r,Q_list,Nion))     #尝试改为np.add(F,Fcoulomb)
    else:
        pass
    if FLaser_True:
        F = np.add(F,Flaser(v))
    else:
        pass
    if FRandom_True and init != True:
        F = np.add(F,Random_Guass_Force())
    else:
        pass
#    a = F / M_list
    a = np.true_divide(F,M_list)
    return a

@jit(float64[:,:]())
def Unit_Vect():
    a = np.random.rand(Nion, 3)-0.5
    a_normal = np.sqrt(np.sum(a*a,axis = 1))
    unit_vector_Li = a/(a_normal.reshape(a_normal.shape[0],1))
    return unit_vector_Li

@jit
def R_O_N_V2_update(r, v, t):
    """Reference: Feng Zhu, PhD,Ion Crystals Produced By Laser and Sympathetic Cooling in a Linear Rf Ion Trap (Mg+)"""
    an = cal_acc(r,v,t)
    rn_12 = r + v*dt/2 + 1/2*an*dt*dt/2/2
    vn_12 = v + 1/2 * an *dt
    an_12 = cal_acc(rn_12,vn_12,t+dt/2)
    r = r + vn_12*dt
    v = v + an_12*dt
    return r, v

@jit
def istrap(r,r0,z0):
    is_no_loss = -r0 < np.max(r[:,0]) < r0 and -r0 < np.max(r[:,1]) < r0 and -z0 < np.max(r[:,2]) < z0
    if is_no_loss == False and init != True:
        print("ion loss")
        r0x = np.abs(r[:,0])
        r1y = np.abs(r[:,1])
        r2z = np.abs(r[:,2])
        r0x_index = np.argwhere(r0x>r0)
        r1y_index = np.argwhere(r1y>r0)
        r2z_index = np.argwhere(r2z>z0)
        is_loss_list = np.append(np.append(r0x_index,r1y_index),r2z_index)
        is_loss_list = np.unique(is_loss_list)
        print(is_loss_list)
        if np.isnan(r[0,0]):
            print("NaN occur")
            print(r)
            is_loss_list = np.array([-1,-1])
        else:
            pass
    else:
        is_loss_list = np.array([])
    return is_loss_list

def Cal_Tem(v,i,average_rf,average_sec,vc_average,vc_square,vc_sec):
    if (i+1) % average_rf == 0:
        vc_average = (vc_average + v[0:N[0],:]) / average_rf
        vc_square = vc_square + np.sum(vc_average**2*M_list[0:N[0],:])
        vc_average = 0
        if (i+1) % (average_rf*average_sec) == 0:
            vc_sec_n = vc_square/average_sec
            vc_square = 0
            return vc_average,vc_square,vc_sec_n
        else:
            pass
            return vc_average,vc_square
    else:
        vc_average = (vc_average + v[0:N[0],:])
        return vc_average,vc_square

"""主程序"""
def Run_Li(r, v,Nstep_run,a_old,break_True):
    xh2, yh2, zh2= zeros(Data_N_index*N[0]), zeros(Data_N_index*N[0]), zeros(Data_N_index*N[0])
    xhLi,yhLi,zhLi = zeros(Data_N_index*N[1]), zeros(Data_N_index*N[1]), zeros(Data_N_index*N[1])
    vc_square = 0
    vc_sec = np.array([])
    vc_average = zeros((N[0],3))
    
    vsy_square = 0
    vsy_average = zeros((N[0],3))
    vsy_sec = np.array([])
    for i in range(Nstep_run):
        r,v = R_O_N_V2_update(r, v, i*dt)
        loss_list = istrap(r,r0,z0)
        if len(loss_list) != 0 and init != True:
            break_True = True
            break
        if init != True:
            if (i+1) % Average_V_Time == 0:
                vc_average = (vc_average + v[0:N[0],:]) / Average_V_Time
                vc_square = vc_square + np.sum(vc_average**2*M_list[0:N[0],:])
                vc_average = 0
                #sympathetic cooling ion
                vsy_average = (vsy_average + v[0:N[0],:]) / Average_V_Time
                vsy_square = vsy_square + np.sum(vsy_average**2*M_list[0:N[0],:])
                vsy_average = 0
                
                if (i+1) % (Average_V_Time*Average_sec_Time) == 0:
                    vc_sec_n = vc_square/Average_sec_Time
                    vc_square = 0
                    vc_sec = np.append(vc_sec,vc_sec_n)
                    #sympathetic cooling ion
                    vsy_sec_n = vsy_square/Average_sec_Time
                    vsy_square = 0
                    vsy_sec = np.append(vsy_sec,vsy_sec_n)
                else:
                    pass
            else:
                vc_average = (vc_average + v[0:N[0],:])
                #sympathetic cooling ion
                vsy_average = (vsy_average + v[0:N[0],:])
            if Nstep - i <= Data_N_index:
                N_index = Data_N_index - (Nstep - i)
                xh2[N_index*N[0]:(N_index+1)*N[0]] = r[0:N[0],0]
                yh2[N_index*N[0]:(N_index+1)*N[0]] = r[0:N[0],1]
                zh2[N_index*N[0]:(N_index+1)*N[0]] = r[0:N[0],2]
                xhLi[N_index*N[1]:(N_index+1)*N[1]] = r[N[0]::,0]
                yhLi[N_index*N[1]:(N_index+1)*N[1]] = r[N[0]::,1]
                zhLi[N_index*N[1]:(N_index+1)*N[1]] = r[N[0]::,2]
            else:
                pass
            if 100 * (i/Nstep_run) % 10 == 0:
                print(100 * (i/Nstep_run) , "%")
            else:
                pass
            break_True = False
            loss_list = []
    if init == True:
        loss_list = []
        print(Nstep_run," setps is calculated")
    return r,v,a_old,break_True,loss_list,i,vc_sec,vsy_sec,xh2, yh2, zh2,xhLi,yhLi,zhLi


"""主程序"""
print("开始初始化")
r_0, v_0,apre_0 = Def_Para.Ini_position_velo()
if init:
    ini_result= Run_Li(r_0, v_0, init_Step, apre_0,True)
    r_0 = ini_result[0]
    para_ion_dynamics = Def_Para.Ini_position_velo()
    v_0 = para_ion_dynamics[1]
    print("完成初始化")
else:
    print("pass")
print("开始主程序")
init = False
start_time = datetime.datetime.now()
print(start_time)
#r,v,a_old,break_True,loss_list,i = Run_Li(r_0, v_0, Nstep, apre_0,True)
def Cycle_ini(r_cy,v_cy,Nstep_cy,apre_0_cy):
    global M_list
    global Q_list
    global Nion
    global N
    break_True = True
    while break_True:
        r_cy, v_cy, a_old,break_True,loss_list,i,vc_sec,vsy_sec,xhc, yhc, zhc,xhsy,yhsy,zhsy = Run_Li(r_cy, v_cy, Nstep_cy, apre_0_cy,break_True)
        if break_True:
            r_cy = np.delete(r_cy,loss_list,axis = 0)
            v_cy = np.delete(v_cy,loss_list,axis = 0)
            apre_0_cy = np.delete(apre_0_cy,loss_list,axis = 0)
            Nstep_cy = Nstep_cy - i
            N[0] = N[0] - np.count_nonzero(loss_list<(N[0]))
            N[1] = N[1] - np.count_nonzero(loss_list>(N[0]-1))
            M_list = np.delete(M_list,loss_list,axis = 0)
            Q_list = np.delete(Q_list,loss_list,axis = 0)
            Nion = N[0] + N[1]
    return r_cy,v_cy,vc_sec,vsy_sec,xhc, yhc, zhc,xhsy,yhsy,zhsy

r_last,v_last,vsec,vsy_sec,xhc, yhc, zhc,xhsy,yhsy,zhsy = Cycle_ini(r_0,v_0,Nstep,apre_0)
print("100%")
print("完成力学积分")
end_time = datetime.datetime.now()
print("结束时间：",end_time)
print("总计算时间",end_time-start_time)

"""参数保存"""
print("###保存参数###")
Int_Method = "R_O_N_V2_update algorithm"
trap_po = "pseudo potential"
if save_csv:
    f = open(Floder_Name + "\\" + "para.txt","a")
    f.write("dt: " + str(dt) + "\n")
    f.write("endtime: " + str(endtime) + "\n")
    f.write("alpha: " + str(alpha) + "\n")
    f.write("MLaser Cooling: " + str(M[0]) + "\n")
    f.write("MSympath Cooling: " + str(M[1]) + "\n")
    f.write("NLaser Cooling: " + str(N[0]) + "\n")
    f.write("NSympath Cooling: " + str(N[1]) + "\n")
    f.write("f: " + str(fre) + "\n")
    f.write("Uend: " + str(Uend) + "\n")
    f.write("Urf: " + str(Urf) + "\n")
    f.write("gfactor_z: " + str(gfactor_z) + "\n")
    f.write("Average_V_Time: " + str(Average_V_Time) + "\n")
    f.write("Temperture_ini_ion: " + str(T_ini_ion) + "\n")
    f.write("beta: " + str(beta) + "\n")
    f.write("speed_incre: " + str(speed_incre) + "\n")
    f.write("Data_N_index: " + str(Data_N_index) + "\n")
    f.write("init_Step: " + str(init_Step) + "\n")
    f.write("Average_sec_Time: " + str(Average_sec_Time) + "\n")
    f.write("integrate Method:" + Int_Method + "\n")
    f.write("Trap Potential:" + trap_po + "\n")
    f.write("All program Time " + str(end_time-start_time) + "\n")
    f.close()
else:
    pass
"""后处理"""
Tion_Pre,T_li,T_list = PlotSaveResult.plot_temper(Floder_Name,vsec,vsy_sec,Average_V_Time,Average_sec_Time,N[0],N[1],dt)
PlotSaveResult.save_temper(Floder_Name,Tion_Pre)
PlotSaveResult.save_ini(Floder_Name,r_0,v_0)
x_sum,y_sum = PlotSaveResult.plot_image(Floder_Name,xhc,yhc,zhc,xhsy,yhsy,zhsy,gamma = 0.5,Ca_Fig = 300, bins = 600,x_y_ratio=1)
PlotSaveResult.enhance_contrast(Floder_Name,"Coling_zy.tif",1.5)

from PIL import Image
from PIL import ImageEnhance
im = Image.open(Floder_Name + "\\" + "Coling_zy.tif")
contrast_origin = ImageEnhance.Contrast(im)
contrast_result = contrast_origin.enhance(1.5)
contrast_result.save(Floder_Name +"\\"+"Coling_zy"+ "contrastup.tif")