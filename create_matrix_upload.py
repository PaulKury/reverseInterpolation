#%%
from statistics import mean
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import scipy
# from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

import seaborn as sns


def create_matrix(csv_name='512_1_10.csv', save=False):
    # Import Data
    file = pd.read_csv(csv_name, delimiter=';')

    # Generate probability lists
    wsk = np.around(np.linspace(0,1,11),decimals=1)

    d = {}
    d['p1'] = []
    d['p2'] = []
    d['p3'] = []

    for p1 in wsk:
        for p2 in wsk:
            for p3 in wsk:
                for n in range(10):
                    d['p1'].append(p1)
                    d['p2'].append(p2)
                    d['p3'].append(p3)

    df_wsk = pd.DataFrame(d)

    p123_FD = pd.concat([df_wsk,file.loc[:,('Db','R2')]], axis=1) # concat dataframes

    p123_FD_avg = p123_FD.groupby(['p1','p2','p3']).agg({'Db':['mean', 'std', 'median']}) # Aggregate (try out with median maybe)
    p123_FD_avg.columns = ['_'.join(col) if col[1] else col[0]for col in p123_FD_avg.columns.values]
    p123_FD_avg.reset_index(inplace=True)

    fig = px.scatter_3d(data_frame=p123_FD_avg, x='p1', y='p2', z='p3',color='Db_mean')
    pio.show(fig)

    # Save results
    if save:
        p123_FD_avg.to_csv('FD_data.csv', index=False, sep=';')
        fig.write_html('FD.html')

    return p123_FD_avg

#%% Reverse Interpolation
def smooth_matrix(m, sigma=1, plot=True, save=False):
    m = m.copy()
    s = m.loc[:,'Db_mean'].to_numpy()
    grid = s.reshape((11,11,11))
    mean_grid = scipy.ndimage.gaussian_filter(grid, sigma)
    values = mean_grid.reshape((1331,))

    if plot:
        m.loc[:,('Db_mean')] = pd.Series(values)
        fig = px.scatter_3d(data_frame=m, x='p1', y='p2', z='p3',color='Db_mean')
        pio.show(fig)
    if (plot and save):
        fig.write_html('FD_smooth.html')

    return values, m 

#%%

def reverse_Interpolation(FD, p123_FD_avg, eps=1e-2, blur=True):
    if blur:
        values = smooth_matrix(p123_FD_avg, sigma=1, plot=False)[0]
    else:
        values = p123_FD_avg.loc[:,('Db_mean')].to_numpy()

    points = p123_FD_avg.loc[:,('p1','p2','p3')].to_numpy()
    linInter= LinearNDInterpolator(points, values)

    # FD-calculation for random guess
    p1, p2, p3 = np.round(np.random.rand(3),3)
    FD_calc = linInter((p1,p2,p3))
    i = 0

    while np.abs(FD_calc - FD) > eps: # execute if the error is bigger than eps

        p1, FD_calc = adj_p1(p1, p2, p3, FD_calc, FD, linInter)
        p2, FD_calc = adj_p2(p1, p2, p3, FD_calc, FD, linInter)
        p3, FD_calc = adj_p3(p1, p2, p3, FD_calc, FD, linInter)

        # print('FD_calc = %g' %FD_calc)
        i +=1

        if i == 10:
            # print('Execute again with different starting value!')
            FD_calc, p1, p2, p3 = reverse_Interpolation(FD, p123_FD_avg, eps=eps, blur=blur)
            return FD_calc, p1, p2, p3

    # print('Number of loops: %g' %i)
    return FD_calc, p1, p2, p3

def adj_p1(p1, p2, p3, FD_calc_prev, FD, linInter):
    old_err = np.abs(FD_calc_prev-FD)

    # Increase p1
    p1_p = p1 + 0.001
    FD_calc_p = linInter((p1_p,p2,p3))
    p_err = abs(FD_calc_p-FD)

    # Decrease p1
    p1_m = p1 - 0.001
    FD_calc_m = linInter((p1_m,p2,p3))
    m_err = abs(FD_calc_m-FD)

    # if (p_err < old_err) and (m_err < old_err):
    #     print('Errors in positive and negative direction got smaller! Potential local minimum!')

    if p_err < old_err: # if new error is smaller
        # print('P1=%g - increase' %p1_p)
        p1, FD_calc_prev = adj_p1(p1_p, p2, p3, FD_calc_p, FD, linInter) # execute function again with updated values

    elif m_err < old_err: # if new error is smaller
        # print('P1=%g - decrease' %p1_m)
        p1, FD_calc_prev = adj_p1(p1_m, p2, p3, FD_calc_m, FD, linInter) # execute function again with updated values

    # elif (p_err > old_err) and (m_err > old_err):
    #     print('Errors in both directions get bigger. Minimum reached!')

    # print('No improvement for P1=%g, stop changing p1.' %p1)
    return p1, FD_calc_prev # only returns if both errors get larger than the previous one; therefore previous error was the smallest

def adj_p2(p1, p2, p3, FD_calc_prev, FD, linInter):
    old_err = np.abs(FD_calc_prev-FD)

    # Increase p1
    p2_p = p2 + 0.001
    FD_calc_p = linInter((p1,p2_p,p3))
    p_err = abs(FD_calc_p-FD)

    # Decrease p1
    p2_m = p2 - 0.001
    FD_calc_m = linInter((p1,p2_m,p3))
    m_err = abs(FD_calc_m-FD)

    # if (p_err < old_err) and (m_err < old_err):
    #     print('Errors in positive and negative direction got smaller! Potential local minimum!')

    if p_err < old_err: # if new error is smaller
        # print('P2=%g - increase' %p2_p)
        p2, FD_calc_prev = adj_p1(p1, p2_p, p3, FD_calc_p, FD, linInter) # execute function again with updated values

    elif m_err < old_err: # if new error is smaller
        # print('P2=%g - decrease' %p2_m)
        p2, FD_calc_prev = adj_p1(p1, p2_m, p3, FD_calc_m, FD, linInter) # execute function again with updated values

    # elif (p_err > old_err) and (m_err > old_err):
    #     print('Errors in both directions get bigger. Minimum reached!')

    # print('No improvement for P2=%g, stop changing p2.' %p2)
    return p2, FD_calc_prev # only returns if both errors get larger than the previous one; therefore previous error was the smallest

def adj_p3(p1, p2, p3, FD_calc_prev, FD, linInter):
    old_err = np.abs(FD_calc_prev-FD)

    # Increase p1
    p3_p = p3 + 0.001
    FD_calc_p = linInter((p1,p2,p3_p))
    p_err = abs(FD_calc_p-FD)

    # Decrease p1
    p3_m = p3 - 0.001
    FD_calc_m = linInter((p1,p2,p3_m))
    m_err = abs(FD_calc_m-FD)

    # if (p_err < old_err) and (m_err < old_err):
    #     print('Errors in positive and negative direction got smaller! Potential local minimum!')

    if p_err < old_err: # if new error is smaller
        # print('P3=%g - increase' %p3_p)
        p3, FD_calc_prev = adj_p1(p1, p2, p3_p, FD_calc_p, FD, linInter) # execute function again with updated values

    elif m_err < old_err: # if new error is smaller
        # print('P3=%g - decrease' %p3_m)
        p3, FD_calc_prev = adj_p1(p1, p2, p3_m, FD_calc_m, FD, linInter) # execute function again with updated values

    # elif (p_err > old_err) and (m_err > old_err):
        # print('Errors in both directions get bigger. Minimum reached!')

    # print('No improvement for P3=%g, stop changing p3.' %p3)
    return p3, FD_calc_prev # only returns if both errors get larger than the previous one; therefore previous error was the smallest

# f = reverse_Interpolation(1.6, m)
# print(f)


