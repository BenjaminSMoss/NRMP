import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid 



def frum_inner(t:float, U:float, U0:float, r:float)->float:
    return np.exp((U-U0-r*t)/0.0256)/(1+np.exp((U-U0-r*t)/0.0256))-t

def calculate_theta(U:np.array, U0:float, r:float)->np.array:
    theta = np.full(len(U), np.nan)
    for n in range(len(U)):
        def Frum_wapper(t):
            return frum_inner(t, U[n], U0, r)
        theta[n] = fsolve(Frum_wapper, 0.5)[0]  # Extract single element
    return theta


def calcualte_all_thetas(U:np.array, U0_l:list=[0.3,0.5,0.8], r_l:np.array=[0,0.2,0.1])->np.array:
    data=pd.DataFrame(index=U)
    for i in range(len(r_l)):
        data[f'U0={U0_l[i]}, r={r_l[i]}']=calculate_theta(U, U0_l[i], r_l[i])
        data.index.name='U'
    return data

def df_deriv(df:pd.DataFrame)->pd.DataFrame:
    dx=pd.DataFrame(df.index.values).diff().values.flatten()
    dy = df.diff(axis=0)
    dydx = pd.DataFrame({col: dy[col] / dx for col in dy.columns})
    return dydx


def Gau(WL, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((WL - mu) / sigma)**2)

def multi_gau(WL:np.array, mu_l:list, sigma_l:list=[100,100,100]):
    y = np.zeros(len(WL))
    for i in range(len(mu_l)):
        y += Gau(WL, mu_l[i], sigma_l[i])
    return y/y.max()

def generate_alpha_bar_spectra(WL:np.array, mu_df:list[list], sigma_df:list[list])->pd.DataFrame:
    data=pd.DataFrame(index=WL)
    for i in range(len(mu_df)):
        data[f'alpha_bar_{i}']=multi_gau(WL, mu_df[i], sigma_df[i])
        data.index.name='WL'
    return data

def generate_SpEC(WL:np.array, U:np.array, alpha_bar_all:pd.DataFrame, theta_all:pd.DataFrame, Q_max_all:list, alpha_max_all:list)->pd.DataFrame:
    # collumn wise multiply alpha_bar_all by alpha_max_all using pd.mul
    alpha_all=alpha_bar_all.mul(alpha_max_all, axis=1)
    # collumn wise multiply theta_all by Q_max_all using pd.mul
    Q_all=theta_all.mul(Q_max_all, axis=1)
    # matrix multiply alpha_all by Q_all
    print(alpha_all.shape, Q_all.T.shape)

    SpEC=pd.DataFrame(np.matmul(np.array(alpha_all), np.array(Q_all.T)))

    SpEC.index=WL
    SpEC.columns=U


    return SpEC

def calculate_Q(all_thetas:pd.DataFrame, Q_max_all:list)->pd.DataFrame:
    return pd.DataFrame(all_thetas.mul(Q_max_all, axis=1), index=all_thetas.index, columns=all_thetas.columns)

def calculate_J_cap(Q_all:pd.DataFrame, scan_rate:float=0.01)->pd.DataFrame:
    return df_deriv(Q_all).multiply(scan_rate)


def calculate_BEP_cat_current(all_thetas:pd.DataFrame, alpha:float=0.5)->pd.DataFrame:
    theta=all_thetas.iloc[:,-1]
    J_cat=np.exp(-13)*theta*np.exp(-(alpha*-0.8*theta)/0.059)
    return pd.DataFrame(J_cat)

def calculate_current(all_thetas:pd.DataFrame, Q_max_all:list, scan_rate:float=0.01)->pd.DataFrame:
    Q_all=calculate_Q(all_thetas, Q_max_all)
    J_cap=pd.DataFrame(calculate_J_cap(Q_all, scan_rate).sum(axis=1))
    J_cat=pd.DataFrame(calculate_BEP_cat_current(all_thetas))

    J_cap.columns=[0]
    J_cat.columns=[0]


    J=np.array(J_cap)+np.array(J_cat)

    return pd.DataFrame(J, index=Q_all.index)

def exp_dec(t:float, a:float, b:float)->float:
    return a*np.exp(-b*t)

def minus_exp_dec(t:float, a:float, b:float)->float:
    return -a*np.exp(-b*t)

def str_exp_dec(t:float, a:float, b:float, c= 0, alpha:float=0.2)->float:
    return -a*(np.exp(-b*t+c)) ** alpha

def linear_decay(t:float, a:float, b:float)->float:
    return a-b*t

def get_nearest_value_to_index(input_df:pd.DataFrame, value:float)->float:
    index_val=(np.abs(input_df.index.values-value)).argmin()
    return input_df.iloc[index_val]


def Q_for_component_in_U_window(comp: int, U1: float, U2: float, Q_all: pd.DataFrame) -> float:
    # get the Q values for the component
    Q = Q_all.iloc[:, comp]
    # get the Q values closest to U1 and U2
    Q_U1 = get_nearest_value_to_index(Q, U1)
    Q_U2 = get_nearest_value_to_index(Q, U2)
    #print(Q_U1, Q_U2)
    # return the sum of Q values in the window

    return np.round((Q_U2 - Q_U1),5)


def piecewise_exp_dec(t: np.ndarray, a: float, b: float, t1: float, t2: float) -> np.ndarray:
    J = np.zeros_like(t, dtype=float)
    J[t <= t1] = 0
    J[(t > t1) & (t <= t2)] = exp_dec(t[(t > t1) & (t <= t2)] - t1, a, b)
    J[t > t2] = minus_exp_dec(t[t > t2] - t2, a, b)
    return J

def piecewise_exp_rise_str_exp_dec(t: np.ndarray, a: float, a1:float, b: float, b1:float, t1: float, t2: float) -> np.ndarray:
    J = np.zeros_like(t, dtype=float)
    J[t <= t1] = 0
    J[(t > t1) & (t <= t2)] = exp_dec(t[(t > t1) & (t <= t2)] - t1, a, b)
    J[t > t2] = str_exp_dec(t[t > t2] - t2, a1, b1)
    #J[t > t2]=np.where(J[t > t2]>0, J, 0)
    return J






def piecewise_exp_dec_integrated(t: np.ndarray, a: float, b: float, t1: float, t2: float) -> np.ndarray:
    """numerically integrate the piecewise exponential decay function"""
    J = np.zeros_like(t, dtype=float)
    J=piecewise_exp_dec(t, a, b, t1, t2)
    return cumulative_trapezoid(J, t, initial=0)

def piecewise_exp_rise_str_exp_dec_integrated(t: np.ndarray, a: float, a1:float, b: float, b1:float, t1: float, t2: float) -> np.ndarray:
    """numerically integrate the piecewise exponential decay function"""
    J = np.zeros_like(t, dtype=float)
    J=piecewise_exp_rise_str_exp_dec(t, a, a1, b, b1, t1, t2)
    Q = cumulative_trapezoid(J, t, initial=0)
    #Q=np.where(Q<0,0,Q)
    return Q



def max_PW_exp_dec_integrated(t: np.ndarray, a: float, b: float, t1: float, t2: float) -> float:
    return piecewise_exp_dec_integrated(t, a, b, t1, t2).max()

def max_piecewise_exp_rise_str_exp_dec_integrated(t: np.ndarray, a: float, a1:float, b: float, b1:float, t1: float, t2: float) -> float:
    return piecewise_exp_rise_str_exp_dec_integrated(t, a, a1, b, b1, t1, t2).max()


def calculate_a_for_Q(Q:float, t:float,  b:float, t1:float, t2:float,a_start:float=0)->float:
    def pw_max_wrapper(a):
        return max_PW_exp_dec_integrated(t, a, b, t1, t2)-Q
    return fsolve(pw_max_wrapper, a_start)[0]


def get_nearest_spectrum_to_U(SpEC:pd.DataFrame, U:float)->pd.array:
    index_val=(np.abs(SpEC.columns.values-U)).argmin()
    return SpEC.iloc[:,index_val]  

def get_diff_U1_U2(SpEC:pd.DataFrame, U1:float, U2:float)->pd.array:
    SpEC_U1=get_nearest_spectrum_to_U(SpEC, U1)
    SpEC_U2=get_nearest_spectrum_to_U(SpEC, U2)
    return SpEC_U2-SpEC_U1

def PW_J_for_U1_U2(U1:float, U2:float, Q_all:pd.DataFrame, t:np.array, b:float, t1:float, t2:float)->pd.DataFrame:
    """This function 1. calculates Q for U1 and U2 in Q_all.sum
    2. calculates a for Q1 and Q2"""
    Q=0
    for i in range(Q_all.shape[1]):
        Q+= Q_for_component_in_U_window(i, U1, U2, Q_all)
    
    a=calculate_a_for_Q(Q=Q, t=t, b=b, t1=t1, t2=t2)

    J=piecewise_exp_dec(t, a, b, t1, t2)

    return pd.DataFrame(J, index=t)

def calculate_PD_dynamics_at_U2(U2:float, U1:float, all_thetas:pd.DataFrame, Q_all:pd.DataFrame, t:np.array, b:float=0.5, b1:float=0.1, t1:float=10, t2:float=60, plotbool:bool=False)->tuple[pd.DataFrame]:
    J=calculate_BEP_cat_current(all_thetas)
    JBEP=get_nearest_value_to_index(J, U2).values[0]
    # fraction_to_decay=0.5
    Q_step=Q_for_component_in_U_window(Q_all.shape[1]-1, U1, U2, Q_all)
    # print(f'U2 is {U2}, JBEP is {JBEP}, Qstep is {Q_step}')
    a1t=JBEP
    a_Q=calculate_a_for_Q(Q=Q_step,t=t,b=1,t1=t1,t2=t2)
    # print(f'a_Q is {a_Q}')
    PDJt=piecewise_exp_rise_str_exp_dec(t=t, a=a_Q, a1=a1t,  b=b, b1=b1,  t1=t1, t2=t2)
    PDQt=piecewise_exp_rise_str_exp_dec_integrated(t=t, a=a_Q, a1=a1t,  b=b, b1=b1, t1=t1, t2=t2)
    if plotbool:
        fig, ax = plt.subplots(2,1 )
        ax[0].plot(t, PDJt)
        ax[1].plot(t, PDQt)
        #add horizontal line at JBEP
        ax[0].axhline(y=-1*JBEP, color='r', linestyle='--')
    return t, PDQt, PDJt


    #a1=test_a1_for_Q(Q=Q,t=t,b1=b1,a_1_start=0, t2=t2)
    
 

    print(f'Q is {Q},a is {a},a1 is {a1}, b1 is {b1}')

    J=piecewise_exp_rise_str_exp_dec(t=t, a=a, a1=a1, b=b, b1=b1, t1=t1, t2=t2)

    Q_rise=J[t<t2].cumsum()[-1]
    Q_fall=J[t>t2].cumsum()[-1]

    print(f'riseQ {Q_rise}, fallQ={Q_fall}')


    return J



def SW_SpEC(SpEC: pd.DataFrame, U1:float, U2:float, Q_all:pd.DataFrame, t:np.array, b:float, t1:float, t2:float)->pd.DataFrame:
    """This function 1. calculates the difference between SpEC for U1 and U2
    2. calculates the PW_J for U1 and U2
    3. Uses np.outer to calculate the SW_SpEC time evolution"""
    SpEC_diff=pd.DataFrame(get_diff_U1_U2(SpEC, U1, U2))
    J_t=PW_J_for_U1_U2(U1, U2, Q_all, t, b, t1, t2)
    Q_t=cumulative_trapezoid(J_t.values.flatten(), t, initial=0)
    Q_t_n=(Q_t-Q_t.min())/(Q_t.max()-Q_t.min()) # as we already have the magnitude of the difference in SpEC, we can normalize the Q_t
    # so all we need to know is the evolution of Delta Q. 

    SW_SpEC=np.outer(SpEC_diff, Q_t_n)
    return pd.DataFrame(SW_SpEC, index=SpEC.index, columns=t)

def PD_SpEC(SpEC:pd.DataFrame,U1:float, U2:float, all_thetas:pd.DataFrame, Q_all:pd.DataFrame, t:np.array, b:float, b1:float, t1:float, t2:float)->pd.DataFrame:
    """This function 1. calculates the difference between SpEC for U1 and U2
    2. calculates the PD_J for U1 and U2
    3. Uses np.outer to calculate the PD_Spec time evolution"""
    SpEC_diff=pd.DataFrame(get_diff_U1_U2(SpEC, U1, U2))
    t, PDQt, PDJt=calculate_PD_dynamics_at_U2(U2, U1, all_thetas, Q_all, t, b, b1, t1, t2)
    PDQt_n=(PDQt-PDQt.min())/(PDQt.max()-PDQt.min())
    PD_Spec=np.outer(SpEC_diff, PDQt_n)
    return pd.DataFrame(PD_Spec, index=SpEC.index, columns=t)


def intrinsic_rate(Q_all:pd.DataFrame, J_BEP:pd.DataFrame)->pd.DataFrame:
    J_BEP.columns=[0]
    Qf=pd.DataFrame(Q_all.iloc[:,-1])
    Qf.columns=[0]
    return J_BEP.divide(Qf,axis=0)




