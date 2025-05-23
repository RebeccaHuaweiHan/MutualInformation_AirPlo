import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
import seaborn as sb
from scipy.stats import pearsonr
from scipy.interpolate import interp1d, lagrange
xlength = 3.27*2.2
ylength = 3.27 * 0.75*2.2
config = {
    "font.family": 'serif',  # 'Times New Roman'
    # "font.serif": ['simsun'],
    "font.serif": ['Times New Roman'],
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "axes.unicode_minus": False,  # 用来正常显示负号
    "figure.figsize": (xlength, ylength),
    "xtick.direction": 'in',
    "ytick.direction": 'in',
}
plt.rcParams.update(config)

# file path of air pollutants and temperature
filepath = "./Toronto_ON_2000_2021.xlsx"
filepath_tempt = "./temperature_Toronto_ON_2000_2021.xlsx"
n = 8036 # both datasets have 8036 rows
def read_temperature(path, n_row):
    """
    读取温度数据
    """
    df_tempt = pd.read_excel(path, nrows=n_row,usecols=[13])
    df_tempt.interpolate(method='polynomial', order=2, inplace=True)
    df_tempt = df_tempt['Mean Temp (°C)']
    print(df_tempt,type(df_tempt))
    return df_tempt

# 读取filepath中Toronto的8类污染物数据
df_CO = pd.read_excel(filepath, sheet_name='CO', nrows=n, usecols=list(range(7,31)))
df_NO = pd.read_excel(filepath, sheet_name='NO', nrows=n, usecols=list(range(7,31)))
df_NO2 = pd.read_excel(filepath, sheet_name='NO2', nrows=n, usecols=list(range(7,31)))
df_NOX = pd.read_excel(filepath, sheet_name='NOX', nrows=n, usecols=list(range(7,31)))
df_O3 = pd.read_excel(filepath, sheet_name='O3', nrows=n, usecols=list(range(7,31)))
df_PM10 = pd.read_excel(filepath, sheet_name='PM10', nrows=n, usecols=list(range(7,31)))
df_PM25 = pd.read_excel(filepath,sheet_name='PM25', nrows=n, usecols=list(range(7,31)))
df_SO2 = pd.read_excel(filepath,sheet_name='SO2', nrows=n, usecols=list(range(7,31)))


cols = list(df_O3.columns)
r, c = df_O3.shape
def row_mean(row):
    avg = row.mean()
    for col in cols:
        if np.isnan(row[col]):
            row[col] = avg
    return row

def df_daily_mean(df):
    """
    求取原始数据的24-hour mean
    """
    df[df < 0] = None
    null_count = df.isnull().sum().sum()
    p_hourly = null_count/(r*c)
    print("Hourly missing%:",p_hourly)
    # use mean value to interpolate hourly data with at leas 1 hour not -999
    df[cols] = df.apply(row_mean, axis=1, result_type='expand')
    # calculate the daily mean
    daily_mean = df.mean(1) # df.isnan(df.mean(1))?
    null_daily_count = daily_mean.isnull().sum().sum()
    print(null_daily_count,r)
    p_daily = null_daily_count/r
    print("Daily missing%:", p_daily)
    daily_mean.interpolate(method='polynomial',order=2,inplace=True)
    # for daily_mean, replace the minus value with 0
    for i in range(r):
        if daily_mean[i]<=0:
            daily_mean[i]=daily_mean[i-1]
    print(daily_mean.isnull().sum())
    print(daily_mean,type(daily_mean))
    daily_mean = daily_mean.round(decimals=1)
    return daily_mean

def output_all_variables():
    """
    计算24-hour mean并将8类air pollutants写入一个xlsx文件，便于后续操作
    """
    daily_mean_temperature = read_temperature(path = filepath_tempt, n_row = n)
    daily_mean_CO = df_daily_mean(df_CO)
    daily_mean_NO = df_daily_mean(df_NO)
    daily_mean_NO2 = df_daily_mean(df_NO2)
    daily_mean_NOX = df_daily_mean(df_NOX)
    daily_mean_O3 = df_daily_mean(df_O3)
    daily_mean_PM10 = df_daily_mean(df_PM10)
    daily_mean_PM25 = df_daily_mean(df_PM25)
    daily_mean_SO2 = df_daily_mean(df_SO2)
    df = pd.DataFrame({'tempt':daily_mean_temperature,
                       'CO':daily_mean_CO,
                       'NO':daily_mean_NO,
                       'NO2':daily_mean_NO2,
                       'NOX':daily_mean_NOX,
                       'O3':daily_mean_O3,
                       'PM25':daily_mean_PM25,
                       'SO2':daily_mean_SO2,
                       'PM10':daily_mean_PM10})
    df.to_excel('pollutants_2000_2021.xlsx',sheet_name='pollutants',index=True)

def multi_plot():
    """
    plot 8 air pollutants and the relations of several of them
    """
    filepath = "./pollutants_2000_2021.xlsx"
    df = pd.read_excel(filepath, sheet_name='pollutants', nrows=8036)
    tempt = df['tempt']
    CO = df['CO']
    NO = df['NO']
    NO2 = df['NO2']
    NOX = df['NOX']
    O3 = df['O3']
    SO2 = df['SO2']
    PM25 = df['PM25']

    font_size = 10
    # figure size configuration
    x = n
    mk_size =0.8
    l = 0.13
    r = 0.92
    t = 0.92
    b = 0.1
    h = 0.3
    # w = 0.17
    w = 0.1
    lw =0.5
    color_list = ['#808080', '#4682B4', '#FFA500', '#A52A2A', '#00CED1', '#008000', '#800080','#FFD700']
    ##########################################################
    # 画8个air pollutants和时间的关系
    fig,ax = plt.subplots(4,2)
    ax[0,0].set_xlim(0, x)
    ylim_min = np.around(np.amin(tempt)*1.1,decimals=1)
    ylim_max = np.around(np.amax(tempt)*1.1,decimals=1)
    ax[0,0].set_ylim(ylim_min, ylim_max)
    ax[0,0].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    # ax[0,0].set_yticks(np.arange(-25, 35.1, 20), fontproperties='Times New Roman', size=font_size)
    ax[0,0].set_yticks(np.arange(-25, 35.1, 20))
    ax[0, 0].set_ylabel('T/ °C', size=font_size, labelpad=1)  # 输入乘号？
    # ax[0,0].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[0,0].plot(np.arange(x), tempt, 'o-', color = color_list[0],linewidth=lw, markersize=mk_size)
    # ax[0,0].legend(loc = 'upper right',frameon = False,  prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[0,1].set_xlim(0, x)
    ylim_min = np.around(np.amin(CO)*1.1,decimals=1)
    ylim_max = np.around(np.amax(CO)*1.1,decimals=1)
    ax[0,1].set_ylim(ylim_min, ylim_max)
    ax[0,1].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    # ax[0,1].set_yticks(np.arange(0, 4.1, 1), fontproperties='Times New Roman', size=font_size)
    ax[0,1].set_yticks(np.arange(0, 4.1, 1))
    # ax[0,1].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[0,1].set_ylabel('CO/ ppm', size=font_size, labelpad=1) # 输入乘号？
    ax[0,1].plot(np.arange(x), CO, 'o-', color = color_list[1],linewidth=lw, markersize=mk_size)
    # ax[0,1].legend(loc = 'upper right', frameon = False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[1, 0].set_xlim(0, x)
    ylim_min = np.around(np.amin(NO) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(NO) * 1.1, decimals=1)
    ax[1, 0].set_ylim(ylim_min, ylim_max)
    ax[1, 0].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[1, 0].set_yticks(np.arange(0, 233, 58))
    # ax[1, 0].set_yticks(np.arange(0, 233, 58), fontproperties='Times New Roman', size=font_size)
    # ax[1, 0].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[1, 0].set_ylabel('NO/ ppb', size=font_size, labelpad=1)  # 输入乘号？
    ax[1, 0].plot(np.arange(x), NO, 'o-', color = color_list[2],linewidth=lw, markersize=mk_size, label='NO')
    # ax[1, 0].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[1, 1].set_xlim(0, x)
    ylim_min = np.around(np.amin(NO2) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(NO2) * 1.1, decimals=1)
    ax[1, 1].set_ylim(ylim_min, ylim_max)
    ax[1, 1].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[1, 1].set_yticks(np.arange(0, 72.1, 18))
    # ax[1, 1].set_yticks(np.arange(0, 72.1, 18),fontproperties='Times New Roman', size=font_size)
    # ax[1, 1].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[1, 1].set_ylabel('$\mathrm{NO_2}$/ ppb', size=font_size, labelpad=1)  # 输入乘号？
    ax[1, 1].plot(np.arange(x), NO2, 'o-', color = color_list[3],linewidth=lw, markersize=mk_size, label='NO2')
    # ax[1, 1].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[2, 0].set_xlim(0, x)
    ylim_min = np.around(np.amin(NOX) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(NOX) * 1.1, decimals=1)
    ax[2, 0].set_ylim(ylim_min, ylim_max)
    ax[2, 0].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[2, 0].set_yticks(np.arange(0, 284.1, 71))
    # ax[2, 0].set_yticks(np.arange(0, 284.1, 71), fontproperties='Times New Roman', size=font_size)
    # ax[2, 0].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[2, 0].set_ylabel('$\mathrm{NO_X}$/ ppb', size=font_size, labelpad=1)  # 输入乘号？
    ax[2, 0].plot(np.arange(x), NOX, 'o-', color = color_list[4],linewidth=lw, markersize=mk_size, label='NOX')
    # ax[2, 0].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[2, 1].set_xlim(0, x)
    ylim_min = np.around(np.amin(O3) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(O3) * 1.1, decimals=1)
    ax[2, 1].set_ylim(ylim_min, ylim_max)
    ax[2, 1].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[2, 1].set_yticks(np.arange(0, 72.1, 18))
    # ax[2, 1].set_yticks(np.arange(0, 72.1, 18),fontproperties='Times New Roman', size=font_size)
    # ax[2, 1].set_xlabel(r'Day', size=font_size, labelpad=1)
    ax[2, 1].set_ylabel('$\mathrm{O_3}$/ ppb', size=font_size, labelpad=1)  # 输入乘号？
    ax[2, 1].plot(np.arange(x), O3, 'o-', color = color_list[5],linewidth=lw, markersize=mk_size, label='O3')
    # ax[2, 1].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[3, 0].set_xlim(0, x)
    ylim_min = np.around(np.amin(PM25) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(PM25) * 1.1, decimals=1)
    ax[3, 0].set_ylim(ylim_min, ylim_max)
    ax[3, 0].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[3, 0].set_yticks(np.arange(0, 56.1, 14))
    # ax[3, 0].set_yticks(np.arange(0, 56.1, 14), fontproperties='Times New Roman', size=font_size)
    ax[3, 0].set_xlabel(r'Year', size=font_size, labelpad=1)
    ax[3, 0].set_ylabel('$\mathrm{PM_{2.5}}$/ $\mathrm{\mu g/m^{3}}$', size=font_size, labelpad=1)  # 输入乘号？
    ax[3, 0].plot(np.arange(x), PM25, 'o-', color = color_list[6], linewidth=lw, markersize=mk_size, label='PM25')
    # ax[3, 0].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    ##################################
    ax[3, 1].set_xlim(0, x)
    ylim_min = np.around(np.amin(SO2) * 1.1, decimals=1)
    ylim_max = np.around(np.amax(SO2) * 1.1, decimals=1)
    ax[3, 1].set_ylim(ylim_min, ylim_max)
    ax[3, 1].set_xticks(np.arange(0, 8036, 2555), [r'2000', r'2007', r'2014', r'2021'], size=font_size)
    ax[3, 1].set_yticks(np.arange(0, 24.1, 6))
    # ax[3, 1].set_yticks(np.arange(0, 24.1, 6), fontproperties='Times New Roman', size=font_size)
    ax[3, 1].set_xlabel(r'Year', size=font_size, labelpad=1)
    ax[3, 1].set_ylabel('$\mathrm{SO_2}$/ ppb', size=font_size, labelpad=1)  # 输入乘号？
    ax[3, 1].plot(np.arange(x), SO2, 'o-', color = color_list[7],linewidth=lw, markersize=mk_size, label='SO2')
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    # ax[3, 1].legend(loc='upper right', frameon=False, prop={'family': 'Times New Roman', 'size': font_size})
    plt.savefig('pollutants.png', dpi=300,  bbox_inches='tight', transparent=True) #指定分辨率保存
    ##########################################################
    #画几个air pollutants的关系
    fig_size= (xlength, ylength*1.1)
    fig1, ax1 = plt.subplots(4,figsize=fig_size)

    lpx = 0.05
    lpy = 1
    ax1[0].set_xlabel(r'$\mathrm{NO_X}$/ ppb', size=font_size, labelpad=lpx)
    ax1[0].set_ylabel(r'NO/ ppb', size=font_size, labelpad=lpy)  # 输入乘号？
    ax1[0].plot(NOX, NO, 'o', color = color_list[4],linewidth=lw, markersize=mk_size, label='SO2')

    ax1[1].set_xlabel(r'T/ °C', size=font_size, labelpad=lpx)
    ax1[1].set_ylabel(r'$\mathrm{O_3}$/ ppb', size=font_size, labelpad=lpy)  # 输入乘号？
    ax1[1].plot(tempt, O3,  'o', color=color_list[5], linewidth=lw, markersize=mk_size, label='SO2')

    ax1[2].set_xlabel(r'$\mathrm{PM_{2.5}}$/ $\mathrm{\mu g/m^{3}}$', size=font_size, labelpad=lpx)
    ax1[2].set_ylabel(r'$\mathrm{O_3}$/ ppb', size=font_size, labelpad=lpy)  # 输入乘号？
    ax1[2].plot(PM25, O3, 'o', color=color_list[1], linewidth=lw, markersize=mk_size, label='SO2')
    # ax1[3].plot(PM25, NO2, 'o', color=color_list[7], linewidth=lw, markersize=mk_size, label='SO2')
    ax1[3].set_xlabel(r'CO/ ppb', size=font_size, labelpad=lpx)
    ax1[3].set_ylabel(r'$\mathrm{NO_2}$/ ppb', size=font_size, labelpad=lpy)  # 输入乘号？
    ax1[3].plot(NO2, CO,'o', color=color_list[7], linewidth=lw, markersize=mk_size, label='SO2')
    # ax1[4].plot(O3, CO, 'o', color=color_list[7], linewidth=lw, markersize=mk_size, label='SO2')
    # ax1[5].plot(O3, PM25, 'o', color=color_list[7], linewidth=lw, markersize=mk_size, label='SO2')

    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace= 0.38, wspace = w)
    plt.savefig('correlation.png', dpi=300,  bbox_inches='tight', transparent=True) #指定分辨率保存


def mic_pearson():
    """
    计算MIC和PIC，分别使用全部数据、cold season数据和warm season数据
    :return: MIC, PC, MIC_w, PC_w, MIC_c, PC_c
    """

    # from sklearn.feature_selection import mutual_info_classif
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    filepath = "./pollutants_2000_2021.xlsx"
    n =8036
    df = pd.read_excel(filepath, sheet_name='pollutants', nrows=n, usecols=np.arange(1,9,1).tolist())
    df = df.values
    # 每一年在xlsx文件中的row
    start = [0, 366, 731, 1096, 1461, 1827,2192, 2557, 2922, 3288, 3653, 4018, 4383, 4749, 5114, 5479, 5844, 6210, 6575, 6940, 7305, 7671]
    stop = [365, 730, 1095, 1460, 1826, 2191, 2556, 2921, 3287, 3652, 4017, 4382, 4748, 5113, 5478, 5843, 6209, 6574, 6939, 7304, 7670, 8035]
    #########################################################################
    # # whole pollutants data
    MIC = np.zeros((8,8))
    PC = np.zeros((8,8))
    P_value = np.zeros((8,8))
    for i in np.arange(8):
        if i<=8:
            for j in np.arange(i,8,1):
                mine = MINE(alpha=0.6,c=15)
                mine.compute_score(df[0:,i],df[0:,j])
                MIC[i, j] = mine.mic()
                MIC[j, i] = MIC[i, j]
                PC[i,j],P_value[i,j] = pearsonr(df[0:,i],df[0:,j])
                PC[j,i] = PC[i,j]
    MIC =np.around(MIC, decimals=2)
    PC = np.around(PC, decimals=2)
    ######################################################################
    # # seasonal pollutants data :April-September, October-March
    index_cold, index_warm = get_index()
    # print(type(index_cold))
    # print(len(index_cold),len(index_warm))
    df_cold = df[index_cold, :]
    df_warm = df[index_warm, :]
    # print(df_cold.shape,df_warm.shape)

    MIC_w = np.zeros((8,8))
    PC_w = np.zeros((8,8))
    P_value_w = np.zeros((8,8))
    MIC_c = np.zeros((8, 8))
    PC_c = np.zeros((8, 8))
    P_value_c = np.zeros((8, 8))

    for i in np.arange(8):
        if i<=8:
            for j in np.arange(i,8,1):
                mine = MINE(alpha=0.6,c=15)
                mine.compute_score(df_warm[0:,i],df_warm[0:,j])
                MIC_w[i, j] = mine.mic()
                MIC_w[j, i] = MIC_w[i, j]
                PC_w[i,j], P_value_w[i,j] = pearsonr(df_warm[0:,i],df_warm[0:,j])
                PC_w[j,i] = PC_w[i,j]
    MIC_w =np.around(MIC_w, decimals=2)
    PC_w = np.around(PC_w, decimals=2)
    # print(MIC_w)
    # print(PC_w)
    for i in np.arange(8):
        if i<=8:
            for j in np.arange(i,8,1):
                mine = MINE(alpha=0.6,c=15)
                mine.compute_score(df_cold[0:,i],df_cold[0:,j])
                MIC_c[i, j] = mine.mic()
                MIC_c[j, i] = MIC_c[i, j]
                PC_c[i,j],P_value_c[i,j] = pearsonr(df_cold[0:,i],df_cold[0:,j])
                PC_c[j,i] = PC_c[i,j]
    MIC_c =np.around(MIC_c, decimals=2)
    PC_c = np.around(PC_c, decimals=2)
    return MIC, PC, MIC_w, PC_w, MIC_c, PC_c


def sns_heatmap(x1,x2,text):
    """
    plot the heatmap
    """
    sb.set_theme()
    font = {'family': 'Times New Roman',
            'size': 10,}
    plt.rc('font', family='Times New Roman', size=10)

    axis =['T', 'CO', 'NO', r'$\mathrm{NO_2}$', r'$\mathrm{NO_X}$', r'$\mathrm{O_3}$', \
           r'$\mathrm{PM_{25}}$', r'$\mathrm{SO_2}$']

    f1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=(xlength,ylength*0.6), sharex=True, sharey=True)
    cbar_ax = f1.add_axes([.91, 0.1, .03, .82])
    h1 = sb.heatmap(x1, annot=True, annot_kws=font, ax=ax1[0], cmap='coolwarm',linewidths=0.2, xticklabels=axis, yticklabels=axis, cbar= False)
    ax1[0].set_title('PCC', fontsize=10)
    h2 = sb.heatmap(x2, annot=True, annot_kws=font, ax=ax1[1], cmap='coolwarm',linewidths=0.2, xticklabels=axis, yticklabels=axis, cbar_ax= cbar_ax)
    ax1[1].set_title('MIC', fontsize=10)

    plt.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.1, hspace= 0.01, wspace = 0.15)
    # cbar_ax = f1.add_axes([.91, 0.1, .03, .83])

    plt.savefig(f'{text}.png', dpi=300,  bbox_inches='tight', transparent=True) #指定分辨率保存


def get_index():
    """
    :return: get index of warm days: from April 1 to Sep. 30, cold days: from Oct., 1 to Mar., 31
    """
    index_warm_days = []
    index_cold_days = []
    for i in np.arange(2000, 2022):
        if i ==2000:
            start_cold = 0
            end_cold = from_2000_to_today(year=2000, month=3, day=31)
            index_cold_days = index_cold_days + list(np.arange(start_cold, end_cold+1))
            print(f'cold days of {i}:', list(np.arange(start_cold, end_cold+1)))
        elif i == 2021:
            start_cold = from_2000_to_today(year=i-1, month=10, day=1)
            end_cold = from_2000_to_today(year=i, month=3, day=31)
            index_cold_days= index_cold_days + list(np.arange(start_cold, end_cold + 1))
            start_cold1 = from_2000_to_today(year=2021, month=10, day=1)
            end_cold1 = from_2000_to_today(year=2021, month=12, day=31)
            index_cold_days = index_cold_days + list(np.arange(start_cold1, end_cold1 + 1))
            print(f'cold days of {i}:', list(np.arange(start_cold, end_cold + 1)))
        else:
            start_cold = from_2000_to_today(year=i-1, month=10, day=1)
            end_cold = from_2000_to_today(year=i, month=3, day=31)
            index_cold_days= index_cold_days + list(np.arange(start_cold, end_cold + 1))
            print(f'cold days of {i}:', list(np.arange(start_cold, end_cold + 1)))

        start_warm = from_2000_to_today(year=i, month=4, day=1)
        end_warm = from_2000_to_today(year=i, month=9, day=30)
        index_warm_days = index_warm_days + list(np.arange(start_warm, end_warm + 1))
        print(f'warm days of {i}:', list(np.arange(start_warm, end_warm + 1)))
        # print(i, index_cold_days)
        # print(i, index_warm_days)
    print(len(index_cold_days),len(index_warm_days))
    return index_cold_days, index_warm_days


def from_2000_to_today(year,month,day):
    """
    用get_num_of_day函数提取cold season和warm season的数据
    """
    import calendar
    year = year
    month = month
    day = day
    sum = 0
    if year == 2000:
        days = get_num_of_day(year=year, month=month, day=day)
        return days-1
    else:
        for i in np.arange(2000, year, 1):
            days = 366 if calendar.isleap(int(i)) else 365
            sum = sum + days
            # print(i, year_days, sum)
        days = get_num_of_day(year=year, month=month, day=day)
        sum = sum +days
        # print(sum-1)
        return sum-1


def get_num_of_day(year,month,day):
    """
    输入year，month，day，返回该日期在一年中的天数，用于计算特定日期在xlsx中的行数
    """
    year = year
    month = month
    day = day
    months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 400 == 0 or year % 4 == 0:
        months[2] = months[2] + 1
    if 0 < month <= 12:
        days = 0
        for item in range(month):
            sum = months[item]
            days = days + sum
        day_s = days + day
        # print(f'The number of days is: {day_s}.')
    else:
        print('date out of range!')
    return day_s

if __name__=="__main__":
    # 1.将所有day-mean数据放入一个xlsx文件
    # output_all_variables()
    # 2.画图，下面的函数每个单独运行，速度快点
    multi_plot()
    # 3.计算MIC和PIC
    # MIC, PC, MIC_w, PC_w, MIC_c, PC_c = mic_pearson()
    # 4.画heat map
    # sns_heatmap(PC, MIC, 'correlation')
    # sns_heatmap(PC_w, MIC_w, 'correlation_w')
    # sns_heatmap(PC_c, MIC_c, 'correlation_c')
    plt.show()


