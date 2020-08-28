# -*- coding: utf-8 -*-

__author__='xiangyanfei'

import os
import shutil
import re
import sys
import numpy as np
import pandas as pd
import itertools
import logging
import configparser
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy.interpolate as intl
from scipy.interpolate import griddata
from collections import Counter,OrderedDict
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xyf_utils import *


#Open config file
config = configparser.ConfigParser()
conf_file = open("/share_data/xyf/code/oridata.ini")
config.readfp(conf_file)

en_in_path = config.get("en_path","en_in_path")
en_out_path = config.get("en_path","en_out_path")

hycom_file = config.get("hycom_path","hycom_file")
hycom_in_path = config.get("hycom_path","hycom_in_path")
hycom_out_path = config.get("hycom_path","hycom_out_path")

sla_in_path = config.get("sla_path","sla_in_path")
sla_out_path = config.get("sla_path","sla_out_path")

sss_in_path = config.get("sss_path","sss_in_path")
sss_out_path = config.get("sss_path","sss_out_path")

sst_in_path = config.get("sst_path","sst_in_path")
sst_out_path = config.get("sst_path","sst_out_path")
    
ccmp_in_path = config.get("ccmp_path","ccmp_in_path")
interp_ccmp_out_path = config.get("interp_ccmp_path","interp_ccmp_out_path")    
interp_sla_out_path = config.get("interp_sla_path","interp_sla_out_path")    
    
ccmp_d5_mean_path = config.get("d5_mean_path","ccmp_d5_mean_path")
sla_d5_mean_path = config.get("d5_mean_path","sla_d5_mean_path")
sss_d5_mean_path = config.get("d5_mean_path","sss_d5_mean_path")
sst_d5_mean_path = config.get("d5_mean_path","sst_d5_mean_path")

new_hycom_d5_mean_path = config.get("d5_mean_path","new_hycom_d5_mean_path")


def statistic_all_stations(data_path,output_path):
    """
    统计所有出现过的站点的服役情况：
    1、遍历每年每月下的站点，记录下所有出现的站点
    2、去重，合并重复出现的站点，记录重复出现的次数（如果该站点一直服役，重复出现的次数=60）
    3、将站名、经度、纬度、服役次数作为一行记录保存在csv中

    :param data_path: IOCLevel数据根目录
    :param output_path: csv保存路径
    """
    logging_config(folder='log', name='statistic_all_stations')
    
    org_station_info = []

    year_dirs = sorted(os.listdir(data_path))
    year_dirs.remove('水位数据格式.docx')
    
    # 遍历每年每月的观测站记录，对于其中每一个观测站，获取其name、latitude、longitude，以字典+列表形式记录
    for year_dir in year_dirs:
        logging.info('--------------processing {}---------------'.format(year_dir))
        year_dir_path=os.path.join(data_path,year_dir)

        for month_dir in sorted(os.listdir(year_dir_path)):
            if re.match(r'[0,1]',month_dir) and not month_dir.endswith('rar'):  #查找月份文件夹
                month_dir_path=os.path.join(data_path,year_dir,month_dir)
                obs_list=os.listdir(month_dir_path)
                logging.info('The number of observations:{}'.format(len(obs_list)))
                    
                for obs in obs_list:
                    station_name=obs.split('.')[-1]
                    obs_path = os.path.join(data_path, year_dir, month_dir, obs)
                    
                    # 读取一个观测站记录文件
                    with open(obs_path,'r') as obs_file:
                        obs_file.seek(23)
                        latitude = obs_file.read(6)
                        longitude = obs_file.read(7)
                    
                    station_info_item={'name': station_name, 'latitude': latitude, 'longitude':longitude}
                    org_station_info.append(station_info_item)

    # 去重，以name为键，去除重复记录的站点信息
    station_info = OrderedDict()
    for item in org_station_info:
        # 以name为键，统计同一个站点的服役
        station_info.setdefault(item['name'],{**item,'serve_times':0})['serve_times']+=1
    station_info=list(station_info.values())

    # 保存为csv文件，每行观测站的站名、经度、纬度、服役次数
    df=pd.DataFrame(station_info,columns=['name','latitude','longitude','serve_times'])
    df.to_csv(output_path,index=False)


def statistic_stations_in_service(data_path,info_path):
    """
    统计每年每月下的服役站点：
    1、遍历每年每月下的站点，与所有站点比较，服役的记1，未服役的记0
    2、以’年月‘为列名，将服役站点信息（0/1）作为一列，增加到站点信息中（station_info.csv）
    
    :param data_path: IOCLevel数据根路径
    :param info_path: 所有站点的信息，包括：站名(name)、纬度(latitude)、经度(longitude)、该站点出现次数(serve_times)
    """
    logging_config(folder='log', name='statistic_stations_in_service')
    
    # 获取所有的站点名
    station_df=pd.read_csv(info_path)
    logging.info(station_df)
    
    all_station_name=np.array(station_df['name'].values)
    all_station_name_list=np.squeeze(all_station_name).tolist()
    logging.info(all_station_name_list)

    year_dirs = sorted(os.listdir(data_path))
    year_dirs.remove('水位数据格式.docx')
    
    for year_dir in year_dirs:
        logging.info('--------------processing {}---------------'.format(year_dir))
        year_dir_path=os.path.join(data_path,year_dir)

        # 遍历每年每月每个观测观测记录，每个观测记录记录该月每天的逐时潮高
        for month_dir in sorted(os.listdir(year_dir_path)):
     
            if re.match(r'[0,1]',month_dir) and not month_dir.endswith('rar'):  #查找月份文件夹
                month_dir_path=os.path.join(data_path,year_dir,month_dir)
            
                obs_list=os.listdir(month_dir_path)
                logging.info('The number of observations:{}'.format(len(obs_list)))
                    
                time = year_dir + month_dir
                    
                def get_station_name(obs):
                    station_name=obs.split('.')[-1]
                    return station_name
                    
                station_in_obs=list(map(get_station_name,obs_list))
                # logging.info(station_in_obs)
                    
                # 对于某年某月的观测记录，参照所有的观测站（all_station_name_list），统计其中含有的观测站，有记1，没有记0
                is_exist_list=[station in station_in_obs for station in all_station_name_list]
                is_exist_list=list(map(int,is_exist_list)) # bool True/False --> int 0/1
                station_df[time]=is_exist_list
        
        station_df.to_csv(info_path, index=False)


def write_to_csv(data_path,save_path):
    """
    遍历所有观测记录，记录日期（年月日）、站名、纬度、经度、连续缺失值的个数、24小时的逐时潮高,每一年一个csv文件(2015.csv, 2016.csv, ...)
    :param data_path: IOCLevel数据根目录
    :param save_path: csv文件被保存在该路径下
    """
    logging_config(folder='log', name='write_to_csv')
    
    year_dirs = sorted(os.listdir(data_path))
    year_dirs.remove('水位数据格式.docx')
    
    for year_dir in year_dirs :
        
        # 初始化dataframe
        df = pd.DataFrame(columns = ['time','station_name','latitude','longitude','continuous_missing_max','height'])
    
        year_dir_path=os.path.join(data_path,year_dir)

        # 遍历每年每月每个观测观测记录，每个观测记录记录该月每天的逐时潮高
        for month_dir in sorted(os.listdir(year_dir_path)):
            
            if re.match(r'[0,1]',month_dir) and not month_dir.endswith('rar'):  #查找月份文件夹
                
                month_dir_path=os.path.join(data_path,year_dir,month_dir)
                
                obs_list=os.listdir(month_dir_path)
                logging.info('The number of observations:{}'.format(len(obs_list)))
                    
                for obs in obs_list:
                    # logging.info('Reading observation:{}'.format(obs))
                    station_name=obs.split('.')[-1]
                    obs_path=os.path.join(data_path,year_dir,month_dir,obs)
                       
                    # 读取观测站记录文件
                    with open(obs_path,'r') as f:
                        lines = f.readlines()
                        
                    # 观测站纬度、经度
                    latitude=lines[0][23:29]
                    longitude = lines[0][29:36]
                    # logging.info('lat:{}, long:{}'.format(latitude,longitude))

                    # 1表示00-11时，2表示12-23时，这里将两者合并为一个列表,包含24小时的逐时潮高
                    height_list = []
                    for half_day_obs in lines[1:]:
                        # 00-11时的逐时潮高
                        if half_day_obs[4] == '1':
                            for i in np.arange(5,61,5):
                                 height_list.append(half_day_obs[i:i+4])
                        # 12-23时的逐时潮高
                        elif half_day_obs[4] == '2':
                            day = half_day_obs[2:4]
                            time = year_dir + month_dir + day
                            # logging.info('time:{}'.format(time))
                            for i in np.arange(5,61,5):
                                height_list.append(half_day_obs[i:i+4])
                            # logging.info(len(height_list),height_list)
                            
                            # 统计逐时潮高列表中连续缺失值个数(缺失值:9999)
                            list_same = [list(g) for k,g in itertools.groupby(height_list)]
                            max_missing_count=0
                            for l in list_same:
                                if l[0]=='9999':
                                    # logging.info('至少包含1个连续缺失值（9999）')
                                    if len(l)> max_missing_count:
                                        max_missing_count = len(l)
                            # logging.info('max_missing_count:{}'.format(max_missing_count))
                                
                            # 将日期、站名、纬度、经度、连续缺失值的个数、24小时的逐时潮高作为一行记录插入到dataframe末尾
                            obs_item={'time':time, 'station_name':station_name, 'latitude':latitude, 'longitude':longitude,'continuous_missing_max':max_missing_count, 'height':height_list}
                            df=df.append(obs_item,ignore_index=True)
                            height_list=[] # 清空height_list, 以便记录后一天的逐时潮高
                    # break
        csv_save_path=os.path.join(save_path,year_dir+'.csv')
        logging.info('-------------------------------{} finish---------------------------'.format(year_dir))
        df.to_csv(csv_save_path, index=False)
        # break


def to_lat_10(org_lat):
    """
    将纬度由度分秒转化为十进制
    """
    lat=0
    logging.info(org_lat)
    logging.info(org_lat[:2],org_lat[2:5],org_lat[5])
    if org_lat[5]=='N':
        lat=round(float(org_lat[:2])+float(org_lat[2:5])/600,2)
    elif org_lat[5]=='S':
        lat=-round((float(org_lat[:2])+float(org_lat[2:5])/600),2)
    logging.info('org_lat={},lat={}'.format(org_lat,lat))
    return lat


def to_lon_10(org_lon):
    """
    将经度由度分秒转化为十进制
    """
    lon=0
    logging.info(org_lon[:3],org_lon[3:6],org_lon[6])
    if org_lon[6]=='E':
        lon=round(float(org_lon[:3])+float(org_lon[3:6])/600,2)
    elif org_lon[6]=='W':
        lon=-round((float(org_lon[:3])+float(org_lon[3:6])/600),2)
    logging.info('org_lon={},lon={}'.format(org_lon,lon))
    return lon


def mark_all_stations_on_map(info_path, pic_save_path):
    """
    在地图上标记所有站点
    :param info_path: 站点信息，包括经纬度
    :param pic_save_path: 标记站点的地图
    """

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    
    logging_config(folder='log', name='mark_all_stations_on_map')
    df = pd.read_csv(info_path)

    name_list = list(df['name'].values)
    lat_list = list(df['latitude'].values)
    lon_list = list(df['longitude'].values)
    sv_t_list = list(df['serve_times'].values)
    
    logging.info(len(lat_list), len(lon_list))
    
    # %% 统计没有经纬度信息的站点，并移除这些站点
    missing_loc_list = []
    for name,lat,lon,sv_t in zip(name_list, lat_list, lon_list, sv_t_list):
        if (len(lat)<6 or (lat[5]!='N' and lat[5]!= 'S')) or (len(lon)<7 or (lon[6] != 'E' and lon[6] != 'W')):
            # logging.info('missing loc: {},{}'.format(lat,lon))
            missing_loc_item = {'name':name, 'latitude':lat, 'longitude':lon, 'serve_times':sv_t}
            missing_loc_list.append(missing_loc_item)
            name_list.remove(name)
            lat_list.remove(lat)
            lon_list.remove(lon)
            sv_t_list.remove(sv_t)
    df=pd.DataFrame(missing_loc_list)
    df.to_csv('/share_data/xyf/IOCLevel_Analysis/station_missing_loc.csv', index=False)

    logging.info(missing_loc_list)
    
    # 将经纬度转为10进制
    lat_list=list(map(to_lat_10,lat_list))
    lon_list=list(map(to_lon_10,lon_list))

    # %% 地图绘制
    fig,ax=plt.subplots()

    m = Basemap(ax=ax)

    # 画海岸线
    m.drawcoastlines(color='#A8A8A8',linewidth=0.5)

    # 画大洲，并填充颜色
    m.fillcontinents(color='white',lake_color='lightskyblue')

    # 纬度线，范围为[-90,90],间隔为10
    parallels = np.arange(-90.,90.,20.)
    m.drawparallels(parallels, labels=[False, True, True, False]) # 左 右 上 下

    # 经度线，范围为[-180,180],间隔为20
    meridians = np.arange(-180.,180.,40.)
    m.drawmeridians(meridians,labels=[True, False, False, True])

    ax.scatter(lon_list, lat_list, marker='o', c='red', s=5)

    plt.show()
    plt.savefig(pic_save_path)


def get_height(org_str):
    """
    将原始以字符形式记录的高度转换为float，原始记录方式：4位，第一位为符号+/-
    """
    op = org_str[0]
    if op == '-':
        height = (-1.0) * float(org_str[1:])
    elif op == ' ':
        height = float(org_str[1:].strip())
    else:
        height = float(org_str.strip())
    # print(org_str, height)
    # print(type(height))
    return height


def get_monthly_mean_height_list(IOCLevel_data_dir, year, station_name):
    """
    获取某站点在某一年的12个月的月平均潮高列表
    :param IOCLevel_data_dir:
    :param year: 年份
    :param station_name: 站名 
    """
    logging_config(folder='log', name='get_monthly_mean_height_list')
    
    for y in os.listdir(IOCLevel_data_dir):

        # 找到对应年份的逐时潮高数据
        if y == year+'.csv':
            # print(y)
            df = pd.read_csv(os.path.join(IOCLevel_data_dir,year+'.csv'))

            # 找到对应站点的逐时潮高记录
            station_df = df[df['station_name']== station_name]
            # print(station_df)
            
            # 按月份遍历，分别计算12个月的平均潮高
            month_mean_height_list = []
            for m in np.arange(1,13):
                time = year + str(m).zfill(2)
                logging.info(time)

                # 获取一个月的逐时潮高记录
                station_month_df = station_df.loc[station_df['time'].astype('str').str.contains(time)]
                # print(station_month_df)
                
                month_height_arr = []
                for index,row in station_month_df.iterrows():
                    # print(row['time'])
                    height = row['height'].split(',')
                    
                    h_list = []
                    for h in height:
                        h_list.append(h[2:6])

                    h_list= list(map(get_height, h_list))
                    # print(h_list)
                    month_height_arr.append(h_list)    
                month_height_arr= np.asarray(month_height_arr)
                logging.info(month_height_arr.shape)

                # 计算月平均潮高，保留3位小数
                mean = round(month_height_arr[np.where(month_height_arr!= 9999)].mean(),3)
                month_mean_height_list.append(mean)
            logging.info(month_mean_height_list)
            return month_mean_height_list


def monthly_mean_IOClevel(station_info_path, height_data_dir, monthly_IOClevel_info):
    """
    1、找出一直服役的站点（出现60次）
    2、计算这些站点对应的月平均潮高
         缺失值：忽略，计算该月份其他有效潮高的均值
    3、将这些站点的每年的逐月平均潮高保存在csv中

    :param station_info_path: 所有站点的信息，包括站名、经度、纬度、出现次数
    :param height_data_dir: 按年划分的逐时潮高记录（csv）的文件夹
    :param monthly_IOClevel_info: 逐月平均潮高记录（输出）

    """
    logging_config(folder='log', name='monthly_mean_IOClevel')

    logging.info('****************** find_stations_service_all ******************* ')
    # years = ['2015', '2016', '2017', '2018', '2019']
    years = ['2015','2016','2017', '2018', '2019']

    all_station_df = pd.read_csv(station_info_path)

    # 选出一直服役的站点
    station_serve_all_df=all_station_df[all_station_df['occurrence']==60]
    # logging.info(station_serve_all_df)

    df = pd.DataFrame(columns=['year','station_name','longitude', 'latitude','monthly_mean_height'])
    month_mean_info = []
    for index,row in station_serve_all_df.iterrows():
        station_name = row['name']
        lat = row['latitude']
        lon = row['longitude']
        logging.info('---------------- station_name: {} ------------------'.format(station_name))

        for y in years:
            # 获取某站点某一年（12个月）的月平均潮高
            mean_height_list = get_monthly_mean_height_list(height_data_dir, y, station_name)
            
            for m,h in enumerate(mean_height_list):
                col_name = y + str(m+1).zfill(2)
                print(col_name)
                station_serve_all_df.loc[index, col_name] = h

    station_serve_all_df.to_csv(monthly_IOClevel_info, index = False)


def draw_monthly_mean_curve(monthly_height, pic_save_path):
    """
    随机选取5个站点，绘制其5年逐月月平均潮高曲线
    """

    logging_config(folder='log', name='draw_monthly_mean_curve')

    monthly_height_df = pd.read_csv(monthly_height)
    # 随机抽取n行
    sub_df = monthly_height_df.sample(n=5, replace=True, axis=0)

    station_name_list = sub_df['name'].values
    lon_list = sub_df['longitude'].values
    lat_list = sub_df['latitude'].values
    # 将经纬度转为10进制
    lat_list=list(map(to_lat_10,lat_list))
    lon_list=list(map(to_lon_10,lon_list))

    year_height_df = sub_df.iloc[:,4:]
    time_list = year_height_df.columns
    
    x = np.arange(1,61,1)
    
    colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化
    
    fig = plt.figure(figsize=(10,5))
    
    # 月平均变化曲线
    plt.title('Monthly Mean IOClevel', fontsize=16)
    plt.xlabel('time (year-month)',fontsize=12)
    plt.ylabel('height (cm)',fontsize=12)
    
    for index, name in enumerate(station_name_list):
        h_list = year_height_df.iloc[index].values
        print(index, name, h_list)
        
        #color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
        plt.plot(x, h_list, color=mcolors.TABLEAU_COLORS[colors[index]], linewidth=1, linestyle=':', label=name, marker='o')
    
    # 设置坐标轴刻度
    plt.xticks(x, time_list, rotation = 60, fontsize = 6)
    plt.legend(loc=2)#图例展示位置，数字代表第几象限

    
    # plt.show()#显示图像
    
    mark_station_on_map(lat_list, lon_list, station_name_list, colors, pic_save_path)



def mark_station_on_map(lat_list, lon_list, sta_name, color_list, map_save_path):
    """
    在地图上标记站点
    :param lat_list: 站点纬度（list）
    :param lon_list: 站点经度（list）
    :param sta_name: 站名（list）
    """

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    
    logging_config(folder='log', name='mark_station_on_map')
    
    # %% 地图绘制
    fig,ax=plt.subplots()

    m = Basemap(ax=ax)

    # 画海岸线
    m.drawcoastlines(color='#A8A8A8',linewidth=0.5)

    # 画大洲，并填充颜色
    m.fillcontinents(color='white',lake_color='lightskyblue')

    # 纬度线，范围为[-90,90],间隔为10
    parallels = np.arange(-90.,90.,20.)
    m.drawparallels(parallels, labels=[False, True, True, False]) # 左 右 上 下

    # 经度线，范围为[-180,180],间隔为20
    meridians = np.arange(-180.,180.,40.)
    m.drawmeridians(meridians,labels=[True, False, False, True])

    # colors = ['b','g','r','k','m']
    ax.scatter(lon_list, lat_list, marker = 'o', color = 'r', label='1', s=15)
    
    # 在点旁边添加站点名字
    for index,name in enumerate(sta_name):
        ax.text(lon_list[index],lat_list[index], name, color=mcolors.TABLEAU_COLORS[color_list[index]])

    plt.show()
    # plt.savefig(pic_save_path)


def ccmp_interp_to_hycom(year,hycom_file,in_path,out_path):
    """
    Process original CCMP data and interpolat to hycom, convert the longitude from 0~360 to -180~180
    """
    
    logging_config(folder='log', name='ccmp_interp_to_hycom')
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Read latitude/longitute from HYCOM file
    nc_obj = nc.Dataset(hycom_file)
    lon0 = np.asarray(nc_obj.variables['lon'][:])
    lat0 = np.asarray(nc_obj.variables['lat'][:])
    logging.info('Hycom: lon0:{}, lat0:{}'.format(lon0.shape, lat0.shape)) # lon0:(4500,), lat0:(1564,)

    # Direction of original CCMP observation
    fin0 = sorted(os.listdir(in_path))
    for i in fin0:
        logging.info('------{}-------'.format(i)) 
        
        # 2019两个数据来源
        if(len(os.path.splitext(i)[0].split('_')) == 8):
            time = os.path.splitext(i)[0].split('_')[4]
        if(len(os.path.splitext(i)[0].split('_')) == 7):
            time = os.path.splitext(i)[0].split('_')[3]

        logging.info('file time: {},{}'.format(time,len(time)))

        # 按年处理
        if os.path.splitext(i)[1] == '.nc' and time[:4] == year and len(time) == 8:
            
            outfile = os.path.join(out_path, year, time + '.nc')
            
            # 如果已经处理过则跳过
            if os.path.exists(outfile):
                logging.info('exists {}, skip'.format(outfile))
                continue
            
            nc_obj1 = nc.Dataset(os.path.join(in_path, i))
                    
            lon1 = np.asarray(nc_obj1.variables['longitude'][:])
            lat1 = np.asarray(nc_obj1.variables['latitude'][:])
            t1 = np.asarray(nc_obj1.variables['time'][:])
            uwnd = np.asarray(nc_obj1.variables['uwnd'][:])
            vwnd = np.asarray(nc_obj1.variables['vwnd'][:])
            # logging.info('uwnd exist nan? {}'.format(np.isnan(uwnd).any()))
            # logging.info('vwnd exist nan? {}'.format(np.isnan(vwnd).any()))
            # logging.info('CCMP  lon1:{}, lat1:{}, t1:{}, uwnd:{}, vwnd:{}'.format(lon1.shape, lat1.shape, t1.shape, uwnd.shape, vwnd.shape))
            
            # %% uwnd 数据清洗
            uwnd_max = np.max(uwnd)
            uwnd_min = np.min(uwnd)
            uwnd_val_max = nc_obj1.variables['uwnd'].valid_max
            uwnd_val_min = nc_obj1.variables['uwnd'].valid_min

            if int(time) >= 20160531: # 20160531及之后是_FillValue
                uwnd_fill = nc_obj1.variables['uwnd']._FillValue
            else:
                uwnd_fill = nc_obj1.variables['uwnd']._Fillvalue
            
            uwnd_inval = uwnd[np.where((uwnd > uwnd_val_max) | (uwnd < uwnd_val_min) | (uwnd == uwnd_fill))]
            logging.info('uwnd_inval:{}'.format(uwnd_inval))

            # 处理无效值，将大于valid_max的设为valid_max，小于valid_min的设为valid_min,缺失值设为0
            if len(uwnd_inval) != 0:
                uwnd[uwnd > uwnd_val_max] = uwnd_val_max
                uwnd[uwnd < uwnd_val_min] = uwnd_val_min
                uwnd[uwnd == uwnd_fill] = 0
            logging.info('uwnd:{}'.format(uwnd.shape)) 
            
            # %% vwnd 数据清洗
            vwnd_max = np.max(vwnd)
            vwnd_min = np.min(vwnd)
            vwnd_val_max = nc_obj1.variables['vwnd'].valid_max
            vwnd_val_min = nc_obj1.variables['vwnd'].valid_min

            if int(time) >= 20160531: # 20160531及之后是_FillValue
                vwnd_fill = nc_obj1.variables['vwnd']._FillValue
            else:
                vwnd_fill = nc_obj1.variables['vwnd']._Fillvalue

            vwnd_inval = vwnd[np.where((vwnd > vwnd_val_max) | (vwnd < vwnd_val_min) | (vwnd == vwnd_fill))]
            logging.info('vwnd_inval:{}'.format(vwnd_inval))

            # 处理无效值，将大于valid_max的设为valid_max，小于valid_min的设为valid_min, 缺失值设为0
            if len(vwnd_inval) != 0:
                vwnd[vwnd > vwnd_val_max] = vwnd_val_max
                vwnd[vwnd < vwnd_val_min] = vwnd_val_min
                vwnd[vwnd == vwnd_fill] = 0
            logging.info('vwnd:{}'.format(vwnd.shape)) # (4,628,1440)
            
            # 对4个时间求平均    
            uwnd_mean_4t = np.mean(uwnd, axis=0) # (628, 1440)
            vwnd_mean_4t = np.mean(vwnd, axis=0) # (628, 1440)

            # %% 插值方式一：interp2d(linear) 插值前将hycom的经度+180（-180~180 –> 0~ 360）与当前数据的经度匹配
            lon0 = lon0 + 180 
            newfunc_u = intl.interp2d(lon1, lat1, uwnd_mean_4t, kind='linear')
            uwnd_0 = newfunc_u(lon0, lat0) # (lon1, lat1) ==> (lon0, lat0)
            newfunc_v = intl.interp2d(lon1, lat1, vwnd_mean_4t, kind='linear')
            vwnd_0 = newfunc_v(lon0, lat0)
            logging.info('After interp: vwnd_0:{}, uwnd_0:{}'.format(vwnd_0.shape, uwnd_0.shape))
            logging.info('uwnd_0 exist nan? {}'.format(np.isnan(uwnd_0).any()))
            logging.info('vwnd_0 exist nan? {}'.format(np.isnan(vwnd_0).any()))
           
            # %% 插值方式二：griddata
            # LON1, LAT1 = np.meshgrid(lon1, lat1) # LON1:(628, 1440), LAT1:(628,1440) 
            # logging.info('CCMP: LON1:{}, LAT1:{}'.format(LON1.shape, LAT1.shape))
            # 
            # LON_F = LON1.flatten()
            # LON_F = LON_F - 180
            # LAT_F = LAT1.flatten()
            # UWND_mean_4t_F = uwnd_mean_4t.flatten() # (904320,)
            # VWND_mean_4t_F = vwnd_mean_4t.flatten() # (904320.)
            # logging.info('CCMP: LON_F:{}, LAT_F:{}, UWND_mean_4t_F:{}, VWND_mean_4t_F:{}'.format(LON_F.shape, LAT_F.shape, UWND_mean_4t_F.shape, VWND_mean_4t_F.shape))

            # grid1 = np.dstack((LON_F, LAT_F)) # (1, 904320, 2)
            # logging.info('CCMP: grid1:{}'.format(grid1.shape))
            # # del LON1,LAT1,lon1,lat1
            # del LON1,LAT1
            # 
            # LON0, LAT0 = np.meshgrid(lon0, lat0) # LON0:(1564, 4500) LAT0:(1564, 4500)
            # logging.info('Hycom: LON0:{}, LAT0:{}'.format(LON0.shape, LAT0.shape))

            # uwnd_0 = griddata(grid1[0,:,:], UWND_mean_4t_F, (LON0, LAT0), method = 'linear') 
            # logging.info('After interpolatin, uwnd_0:{}'.format(uwnd_0.shape))
            # vwnd_0 = griddata(grid1[0,:,:], VWND_mean_4t_F, (LON0, LAT0), method = 'linear')
            # logging.info('After interpolatin, vwnd_0:{}'.format(vwnd_0.shape))
            # 
            # logging.info('uwnd_mean_4t_F exist nan? {}'.format(np.isnan(UWND_mean_4t_F).any()))
            # logging.info('vwnd_mean_4t_F exist nan? {}'.format(np.isnan(VWND_mean_4t_F).any()))
            # logging.info('uwnd_0 exist nan? {}'.format(np.isnan(uwnd_0).any()))
            # logging.info('vwnd_0 exist nan? {}'.format(np.isnan(vwnd_0).any()))
            
            # 插值前后对比
            # plt.subplot(221)
            # plt.imshow(uwnd_mean_4t, extent=(np.min(lon1), np.max(lon1), np.min(lat1), np.max(lat1)), origin='lower')
            # plt.title('Original UWND')

            # plt.subplot(222)
            # plt.imshow(uwnd_0, extent=(np.min(lon0), np.max(lon0), np.min(lat0), np.max(lat0)), origin='lower')
            # plt.title('Interp UWND')

            # plt.subplot(223)
            # plt.imshow(vwnd_mean_4t, extent=(np.min(lon1), np.max(lon1), np.min(lat1), np.max(lat1)), origin='lower')
            # plt.title('Original VWND')

            # plt.subplot(224)
            # plt.imshow(vwnd_0, extent=(np.min(lon0), np.max(lon0), np.min(lat0), np.max(lat0)), origin='lower')
            # plt.title('Interp VWND')
            # plt.show()
            # sys.exit(2)
           
            # %%  Move the right side to the left side on the longitude axis, and convert the longitude range from (0, 360) to (-180, 180)
            R_uwnd_0 = uwnd_0[:, int(len(lon0)/2):]
            L_uwnd_0 = uwnd_0[:, :int(len(lon0)/2)]
            new_uwnd_0 = np.concatenate((R_uwnd_0, L_uwnd_0), axis=1)
            del R_uwnd_0, L_uwnd_0
            
            R_vwnd_0 = vwnd_0[:, int(len(lon0)/2):]
            L_vwnd_0 = vwnd_0[:, :int(len(lon0)/2)]
            new_vwnd_0 = np.concatenate((R_vwnd_0, L_vwnd_0), axis=1)
            del R_vwnd_0, L_vwnd_0

            lon0 = lon0 -180
            
            # %% 对比拼接前后
            # plt.subplot(221)
            # plt.imshow(uwnd_mean_4t, extent=(np.min(lon1), np.max(lon1), np.min(lat1), np.max(lat1)), origin='lower')
            # plt.title('Original UWND')

            # plt.subplot(222)
            # plt.imshow(new_uwnd_0, extent=(np.min(lon0), np.max(lon0), np.min(lat0), np.max(lat0)), origin='lower')
            # plt.title('new UWND')

            # plt.subplot(223)
            # plt.imshow(vwnd_mean_4t, extent=(np.min(lon1), np.max(lon1), np.min(lat1), np.max(lat1)), origin='lower')
            # plt.title('Original VWND')

            # plt.subplot(224)
            # plt.imshow(new_vwnd_0, extent=(np.min(lon0), np.max(lon0), np.min(lat0), np.max(lat0)), origin='lower')
            # plt.title('new VWND')
            # plt.show()
            # sys.exit(2)

            # %% save 
            logging.info('====== saving to {} ======'.format(outfile))
            
            # 创建nc文件
            gridspi = nc.Dataset(outfile, 'w', format='NETCDF4')

            # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y)
            gridspi.createDimension('latsize', len(lat0))
            gridspi.createDimension('lonsize', len(lon0))
            gridspi.createDimension('t', 1)
    
            # 创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
            times = gridspi.createVariable('time', np.int32, ('t',))
            latitudes = gridspi.createVariable('latitude', np.float32, ('latsize',))
            longitudes = gridspi.createVariable('longitude', np.float32, ('lonsize',))
            uwnds = gridspi.createVariable('uwnd', np.float32, ('latsize', 'lonsize',))
            vwnds = gridspi.createVariable('vwnd', np.float32, ('latsize', 'lonsize',))
            
            # 增加变量属性--单位
            times.units = 'date'
            latitudes.units ='degree_north'
            longitudes.units = 'degree_east'
            uwnds.units = 'm s-1'
            vwnds.units = 'm s-1'
            
            # 为变量赋值
            times[:] = int(time)
            latitudes[:] = lat0
            longitudes[:]=lon0
            uwnds[:] = new_uwnd_0 
            vwnds[:] = new_vwnd_0

            nc.Dataset.close(gridspi)
            # break

    
def sla_interp_to_hycom(year,hycom_file,in_path,out_path):
    """
    Process original SLA data and interpolat to hycom, convert the longitude from 0~360 to -180~180
    """
    
    logging_config(folder='log', name='sla_interp_to_hycom')
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Read latitude/longitute from HYCOM file
    nc_obj = nc.Dataset(hycom_file)
    lon0 = np.asarray(nc_obj.variables['lon'][:])
    lat0 = np.asarray(nc_obj.variables['lat'][:])
    logging.info('Hycom: lon0:{}, lat0:{}'.format(lon0.shape, lat0.shape)) # lon0:(4500,), lat0:(1564,)

    # Direction of original CCMP observation
    fin0 = sorted(os.listdir(os.path.join(in_path,year)))
    for i in fin0:
        time = os.path.splitext(i)[0].split('_')[5]
        logging.info('------ time: {} ------'.format(time))
        
        if os.path.splitext(i)[1] == '.nc':
            
            out_file = os.path.join(out_path, year, time + '.nc')
            
            # 如果已经处理过则跳过
            if os.path.exists(out_file):
                logging.info('exists {}, skip'.format(out_file))
                continue

            nc_obj1 = nc.Dataset(os.path.join(in_path, year, i))
                    
            lon1 = np.asarray(nc_obj1.variables['longitude'][:])
            lat1 = np.asarray(nc_obj1.variables['latitude'][:])
            sla1 = np.asarray(nc_obj1.variables['sla'][:])
            sla1 = np.squeeze(sla1)
            logging.info('SLA: sla1:{}, lon1:{}, lat1:{}'.format(sla1.shape, lon1.shape, lat1.shape))
           
            # 数据清洗
            logging.info('#invalid:{}'.format(len(sla1[np.where(sla1 == -2147483647)]))) # 433904个无效值
            sla1[sla1==-2147483647] = 0

            # 插值
            newfunc = intl.interp2d(lon1, lat1, sla1, kind='cubic')
            lon0 = lon0 + 180
            sla_0 = newfunc(lon0, lat0)
            logging.info('After interp: sla_0:{}'.format(sla_0.shape))
            logging.info('original sla exist nan? {}'.format(np.isnan(sla1).any()))
            logging.info('Interp sla_0 exist nan? {}'.format(np.isnan(sla_0).any()))

            R_sla_0 = sla_0[:, int(len(lon0)/2):]
            L_sla_0 = sla_0[:, :int(len(lon0)/2)]
            new_sla_0 = np.concatenate((R_sla_0, L_sla_0), axis=1)
            del R_sla_0, L_sla_0
            
            lon0 = lon0 - 180
            
            # %% 对比拼接前后
            # plt.subplot(121)
            # plt.imshow(sla1, extent=(np.min(lon1), np.max(lon1), np.min(lat1), np.max(lat1)), origin='lower')
            # plt.title('Original SLA')

            # plt.subplot(122)
            # plt.imshow(new_sla_0, extent=(np.min(lon0), np.max(lon0), np.min(lat0), np.max(lat0)), origin='lower')
            # plt.title('Interp SLA')
            # plt.show()
            # sys.exit(2)

            # %% save 
            logging.info('====== saving to {} ======'.format(out_file))
            
            gridspi=nc.Dataset(out_file, 'w', format='NETCDF4')

            gridspi.createDimension('t', 1)
            gridspi.createDimension('latsize', len(lat0))
            gridspi.createDimension('lonsize', len(lon0))
    
            times = gridspi.createVariable('time', np.int32, ('t',))
            latitudes = gridspi.createVariable('latitude', np.float32, ('latsize',))
            longitudes = gridspi.createVariable('longitude', np.float32, ('lonsize',))
            sla = gridspi.createVariable('sla', np.float32, ('latsize', 'lonsize',))
            
            times.units = 'date'
            latitudes.units = 'degree_north'
            longitudes.units = 'degree_east'
            sla.units = 'meter'
            
            times[:] = int(time)
            latitudes[:] = lat0
            longitudes[:]=lon0
            sla[:] = new_sla_0

            nc.Dataset.close(gridspi)
    return

def ccmp_d5_mean(year, in_path, out_path):
    """
    Process daily CCMP data and average the 5-day data per month(1~5, 6~10, 11~15, ....)
    """

    logging_config(folder='log', name='ccmp_d5_mean')
    
    d5_start=[1, 6, 11, 16, 21, 26, 32]
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_dir = os.path.join(in_path, year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = os.listdir(data_dir)

    # 按月份遍历
    for m in np.arange(1,13):
        mon = str(m).zfill(2)
        logging.info('month : {}'.format(mon))
        
        # 找出某一月份下的nc文件
        m_files = [files[i] for i,x in enumerate(files) if x.find(year+mon)!=-1]
      
        # 按5天划分
        for i in range(len(d5_start)-1):
            day_split = [str(j).zfill(2) for j in range(d5_start[i], d5_start[i+1])]
            logging.info('day_split:{}'.format(day_split))

            day_cluster = [m_files[j] for j,x in enumerate(m_files) if x[6:8] in day_split]
            logging.info('day_cluster:{}'.format(day_cluster))
            
            # 如果某一时间范围没有数据，跳过·
            if len(day_cluster) == 0:
                continue

            times = [int(x[:8]) for j,x in enumerate(day_cluster)]
            logging.info('times:{}'.format(times))
        
            out_file = os.path.join(out_dir,year+mon+'_'+str(i+1)+'.nc')
            logging.info('out_file: {}'.format(out_file))
            if os.path.exists(out_file):
                continue

            # 对5天的数据求平均，(5, 1564, 4500) ==> (1564, 4500)
            d5_uwnd = []
            d5_vwnd = []
            for index,day in enumerate(day_cluster):
                logging.info('day: {}'.format(day))
                nc_obj = nc.Dataset(os.path.join(in_path, year, day))

                if index == 0:
                    lon = np.asarray(nc_obj.variables['longitude'][:])
                    lat = np.asarray(nc_obj.variables['latitude'][:])
           
                uwnd = np.asarray(nc_obj.variables['uwnd'][:])
                vwnd = np.asarray(nc_obj.variables['vwnd'][:])
                # logging.info('lon:{}, lat:{}, t:{}, uwnd:{}, vwnd:{}'.format(lon.shape, lat.shape, t.shape, uwnd.shape, vwnd.shape))
                d5_uwnd.append(uwnd)
                d5_vwnd.append(vwnd)
            
            # 对5天求平均    
            d5_uwnd_mean = np.mean(np.asarray(d5_uwnd), axis=0)
            d5_vwnd_mean = np.mean(np.asarray(d5_vwnd), axis=0)
            
            # 保存
            logging.info('====== saving to {} ======'.format(out_file))
            out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

            out_nc.createDimension('latsize', len(lat))
            out_nc.createDimension('lonsize', len(lon))
            out_nc.createDimension('t', len(times))
    
            time = out_nc.createVariable('time', np.int32, ('t',))
            latitudes = out_nc.createVariable('latitude', np.float32, ('latsize',))
            longitudes = out_nc.createVariable('longitude', np.float32, ('lonsize',))
            uwnds = out_nc.createVariable('uwnd', np.float32, ('latsize', 'lonsize',))
            vwnds = out_nc.createVariable('vwnd', np.float32, ('latsize', 'lonsize',))
            
            time.units = 'date'
            time.avg_peroid = str(len(times)) +' day'
            latitudes.units ='degree_north'
            longitudes.units = 'degree_east'
            uwnds.units = 'm s-1'
            vwnds.units = 'm s-1'
            
            time[:] = times
            latitudes[:] = lat
            longitudes[:]= lon
            uwnds[:] = d5_uwnd_mean
            vwnds[:] = d5_vwnd_mean

            nc.Dataset.close(out_nc)


def sla_d5_mean(year, in_path, out_path):
    """
    Process daily SLA data and average the 5-day data per month(1~5, 6~10, 11~15, ....)
    """

    logging_config(folder='log', name='sla_d5_mean')
    
    d5_start=[1, 6, 11, 16, 21, 26, 32]
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_dir = os.path.join(in_path, year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = os.listdir(data_dir)

    # 按月份遍历
    for m in np.arange(1,13):
        mon = str(m).zfill(2)
        logging.info('month : {}'.format(mon))
        
        # 找出某一月份下的nc文件
        m_files = [files[i] for i,x in enumerate(files) if x.find(year+mon)!=-1]
      
        # 按5天划分
        for i in range(len(d5_start)-1):
            day_split = [str(j).zfill(2) for j in range(d5_start[i], d5_start[i+1])]
            logging.info('day_split:{}'.format(day_split))

            day_cluster = [m_files[j] for j,x in enumerate(m_files) if x[6:8] in day_split]
            logging.info('day_cluster:{}'.format(day_cluster))

            # 如果某一时间范围没有数据，跳过·
            if len(day_cluster) == 0:
                continue
            
            times = [int(x[:8]) for j,x in enumerate(day_cluster)]
            logging.info('times:{}'.format(times))
        
            out_file = os.path.join(out_dir,year+mon+'_'+str(i+1)+'.nc')
            logging.info('out_file: {}'.format(out_file))
            if os.path.exists(out_file):
                continue

            # 对5天的数据求平均，(5, 1564, 4500) ==> (1564, 4500)
            sla_d5 = []
            for index,day in enumerate(day_cluster):
                logging.info('day: {}'.format(day))
                nc_obj = nc.Dataset(os.path.join(in_path, year, day))

                if index == 0:
                    lon = np.asarray(nc_obj.variables['longitude'][:])
                    lat = np.asarray(nc_obj.variables['latitude'][:])

                sla = np.asarray(nc_obj.variables['sla'][:])
                # logging.info('lon:{}, lat:{}, sla:{}'.format(lon.shape, lat.shape, sla.shape))
                sla_d5.append(sla)

            # 对5天求平均    
            sla_d5_mean = np.mean(np.asarray(sla_d5), axis=0)
            
            # 保存
            logging.info('====== saving to {} ======'.format(out_file))
            out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

            out_nc.createDimension('latsize', len(lat))
            out_nc.createDimension('lonsize', len(lon))
            out_nc.createDimension('t', len(times))
    
            time = out_nc.createVariable('time', np.int32, ('t',))
            latitude = out_nc.createVariable('latitude', np.float32, ('latsize',))
            longitude = out_nc.createVariable('longitude', np.float32, ('lonsize',))
            sla = out_nc.createVariable('sla', np.float32, ('latsize', 'lonsize',))
            
            time.units = 'date'
            time.avg_peroid = str(len(times)) + ' day'
            latitude.units ='degree_north'
            longitude.units = 'degree_east'
            sla.units = 'meter'
            
            time[:] = times
            latitude[:] = lat
            longitude[:]= lon
            sla[:] = sla_d5_mean

            nc.Dataset.close(out_nc)


def sss_d5_mean(year, in_path, out_path):
    """
    Process daily SSS data and average the 5-day data per month(1~5, 6~10, 11~15, ....)
    """

    logging_config(folder='log', name='sss_d5_mean')
    
    d5_start=[1, 6, 11, 16, 21, 26, 32]
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_dir = os.path.join(in_path, 'y'+year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = os.listdir(data_dir)

    # 按月份遍历
    for m in np.arange(1,13):
        mon = str(m).zfill(2)
        logging.info('month : {}'.format(mon))
        
        # 找出某一月份下的nc文件
        m_files = [files[i] for i,x in enumerate(files) if x.find(year+mon)!=-1]
      
        # 按5天划分
        for i in range(len(d5_start)-1):
            day_split = [str(j).zfill(2) for j in range(d5_start[i], d5_start[i+1])]
            logging.info('day_split:{}'.format(day_split))

            day_cluster = [m_files[j] for j,x in enumerate(m_files) if x[6:8] in day_split]
            logging.info('day_cluster:{}'.format(day_cluster))

            # 如果某一时间范围没有数据，跳过·
            if len(day_cluster) == 0:
                continue

            times = [int(x[:8]) for j,x in enumerate(day_cluster)]
            logging.info('times:{}'.format(times))
        
            out_file = os.path.join(out_dir,year+mon+'_'+str(i+1)+'.nc')
            logging.info('out_file: {}'.format(out_file))
            if os.path.exists(out_file):
                continue

            # 对5天的数据求平均，(5, 1564, 4500) ==> (1564, 4500)
            sss_d5 = []
            for index,day in enumerate(day_cluster):
                logging.info('index={}, day:{}'.format(index,day))
                nc_obj = nc.Dataset(os.path.join(in_path,'y'+year, day))
                
                if index == 0:
                    lon = np.asarray(nc_obj.variables['longitude'][:])
                    lat = np.asarray(nc_obj.variables['latitude'][:])
                
                sss = np.asarray(nc_obj.variables['SurfaceSalinity'][:])
                logging.info('lon:{}, lat:{}, sss:{}'.format(lon.shape, lat.shape, sss.shape))
                sss_d5.append(sss)
                    
            # 对5天求平均    
            sss_d5_mean = np.mean(np.asarray(sss_d5), axis=0)
            
            # 保存
            logging.info('====== saving to {} ======'.format(out_file))
            out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

            out_nc.createDimension('latsize', len(lat))
            out_nc.createDimension('lonsize', len(lon))
            out_nc.createDimension('t', len(times))
    
            time = out_nc.createVariable('time', np.int32, ('t',))
            latitude = out_nc.createVariable('latitude', np.float32, ('latsize',))
            longitude = out_nc.createVariable('longitude', np.float32, ('lonsize',))
            sss = out_nc.createVariable('SurfaceSalinity', np.float32, ('latsize', 'lonsize',))
            
            time.units = 'date'
            time.avg_peroid = str(len(times))+' day'
            latitude.units ='degree_north'
            longitude.units = 'degree_east'
            sss.units = 'psu'
            
            time[:] = times
            latitude[:] = lat
            longitude[:]= lon
            sss[:] = sss_d5_mean

            nc.Dataset.close(out_nc)


def sst_d5_mean(year, in_path, out_path):
    """
    Process daily SST data and average the 5-day data per month(1~5, 6~10, 11~15, ....)
    """

    logging_config(folder='log', name='sst_d5_mean')
    
    d5_start=[1, 6, 11, 16, 21, 26, 32]
    
    out_dir = os.path.join(out_path, year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_dir = os.path.join(in_path, 'y'+year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = os.listdir(data_dir)

    # 按月份遍历
    for m in np.arange(1,13):
        mon = str(m).zfill(2)
        logging.info('month : {}'.format(mon))
        
        # 找出某一月份下的nc文件
        m_files = [files[i] for i,x in enumerate(files) if x.find(year+mon)!=-1]
      
        # 按5天划分
        for i in range(len(d5_start)-1):
            day_split = [str(j).zfill(2) for j in range(d5_start[i], d5_start[i+1])]
            logging.info('day_split:{}'.format(day_split))

            day_cluster = [m_files[j] for j,x in enumerate(m_files) if x[6:8] in day_split]
            logging.info('day_cluster:{}'.format(day_cluster))

            # 如果某一时间范围没有数据，跳过·
            if len(day_cluster) == 0:
                continue

            times = [int(x[:8]) for j,x in enumerate(day_cluster)]
            logging.info('times:{}'.format(times))
        
            out_file = os.path.join(out_dir,year+mon+'_'+str(i+1)+'.nc')
            logging.info('out_file: {}'.format(out_file))
            if os.path.exists(out_file):
                continue

            # 对5天的数据求平均，(5, 1564, 4500) ==> (1564, 4500)
            sst_d5 = []
            for index,day in enumerate(day_cluster):
                logging.info('index={}, day:{}'.format(index,day))
                nc_obj = nc.Dataset(os.path.join(in_path,'y'+year, day))
                
                if index == 0:
                    lon = np.asarray(nc_obj.variables['longitude'][:])
                    lat = np.asarray(nc_obj.variables['latitude'][:])
        
                sst = np.asarray(nc_obj.variables['SurfaceTemp'][:])
                # logging.info('lon:{}, lat:{}, sla:{}'.format(lon.shape, lat.shape, sla.shape))

                sst_d5.append(sst)
            
            # 对5天求平均    
            sst_d5_mean = np.mean(np.asarray(sst_d5), axis=0)
            
            # 保存
            logging.info('====== saving to {} ======'.format(out_file))
            out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

            out_nc.createDimension('latsize', len(lat))
            out_nc.createDimension('lonsize', len(lon))
            out_nc.createDimension('t', len(times))
    
            time = out_nc.createVariable('time', np.int32, ('t',))
            latitude = out_nc.createVariable('latitude', np.float32, ('latsize',))
            longitude = out_nc.createVariable('longitude', np.float32, ('lonsize',))
            sst = out_nc.createVariable('SurfaceTemp', np.float32, ('latsize', 'lonsize',))
            
            time.units = 'date'
            time.avg_peroid = str(len(times))+' day'
            latitude.units ='degree_north'
            longitude.units = 'degree_east'
            sst.units = 'kelvin'
            
            time[:] = times
            latitude[:] = lat
            longitude[:]= lon
            sst[:] = sst_d5_mean

            nc.Dataset.close(out_nc)


def convert_hycom_longitude_range(year, in_path):
    """
    convert Hycom longitude range after 20170201: 0~360 ==> -180~180
    
    Move the right side to the left side on longitude axis
    """
    
    logging_config(folder='log', name='convert_hycom_longtitude_ragne')
    
    
    out_dir = os.path.join(in_path, 'new_'+year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_dir = os.path.join(in_path, year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = sorted(os.listdir(data_dir))
    
    for f in files:
        
        out_file = os.path.join(out_dir, f)
        logging.info('out_file: {}'.format(out_file))

        if os.path.exists(out_file):
            logging.info('exist {}'.format(out_file))
            # continue
        
        # 201701的数据不做处理
        if year == '2017' and f.split('_')[0][4:6] == '01':
            logging.info('copy:{}'.format(f))
            shutil.copy(os.path.join(in_path,year,f),out_dir)
            continue

        logging.info('file: {}'.format(f))
        
        nc_obj = nc.Dataset(os.path.join(in_path, year, f))

        lon = np.asarray(nc_obj.variables['lon'][:]) 
        
        # convert longitude from [0,360] to [-180,180]
        new_lon = lon - 180
       
        lat = np.asarray(nc_obj.variables['lat'][:])
        
        time = np.asarray(nc_obj.variables['time'][:])

        depth = np.asarray(nc_obj.variables['depth'][:])
        
        temp = np.asarray(nc_obj.variables['t'][:])
        logging.info('temperature: {}'.format(temp.shape))

        sal = np.asarray(nc_obj.variables['s'][:])
        logging.info('salinity: {}'.format(sal.shape))

        # %%  Move the right side of longitude to the left side
        R_temp = temp[:,:,int(len(new_lon)/2):]
        
        L_temp = temp[:,:,:int(len(new_lon)/2)]

        new_temp = np.concatenate((R_temp,L_temp),axis=2)
        
        del R_temp, L_temp

        R_sal = sal[:,:,int(len(new_lon)/2):]
        
        L_sal = sal[:,:,:int(len(new_lon)/2)]

        new_sal = np.concatenate((R_sal, L_sal),axis=2)
        
        del R_sal, L_sal

        # %% 对比拼接前后
        plt.subplot(221)
        plt.imshow(temp[0], extent=(np.min(lon), np.max(lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('Original Temperature')

        plt.subplot(222)
        plt.imshow(new_temp[0], extent=(np.min(new_lon), np.max(new_lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('new Temperature')

        plt.subplot(223)
        plt.imshow(sal[0], extent=(np.min(lon), np.max(lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('Original Salinity')

        plt.subplot(224)
        plt.imshow(new_sal[0], extent=(np.min(new_lon), np.max(new_lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('new Salinity')
        plt.show()
        sys.exit(2)

        # %% save to a new nc
        logging.info('====== saving to {} ======'.format(out_file))
        out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

        out_nc.createDimension('latsize', len(lat))
        out_nc.createDimension('lonsize', len(new_lon))
        out_nc.createDimension('depth', len(depth))
        out_nc.createDimension('t', len(time))

        times = out_nc.createVariable('time', np.int32, ('t',))
        latitude = out_nc.createVariable('lat', np.float32, ('latsize',))
        longitude = out_nc.createVariable('lon', np.float32, ('lonsize',))
        dep = out_nc.createVariable('depth', np.float32, ('depth',))
        temp = out_nc.createVariable('t', np.float32, ('depth', 'latsize','lonsize'))
        sal = out_nc.createVariable('s', np.float32, ('depth', 'latsize','lonsize'))

        
        times[:] = time
        latitude[:] = lat
        longitude[:]= new_lon
        dep[:] = depth
        temp[:] = new_temp
        sal[:] = new_sal

        nc.Dataset.close(out_nc)
        
       # break


def convert_hycom_d5_mean_lon_range(year, in_path, out_path):
    """
    convert Hycom_d5_mean longitude range after 201702_1: 0~360 ==> -180~180
    
    Move the right side to the left side on longitude axis
    """

    logging_config(folder='log', name='convert_hycom_d5_mean_lon_ragne')
    
    out_dir = os.path.join(out_path, 'new_'+year)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_dir = os.path.join(in_path, year)
    logging.info('-------------- Processing {} ---------------'.format(data_dir))

    files = sorted(os.listdir(data_dir))
    
    for f in files:
        
        out_file = os.path.join(out_dir, f)
        logging.info('out_file: {}'.format(out_file))

        if os.path.exists(out_file):
            logging.info('exist {}'.format(out_file))
            # continue
        
        # 201701的数据不做处理
        if year == '2017' and f.split('_')[0][4:6] == '01':
            logging.info('copy:{}'.format(f))
            shutil.copy(os.path.join(in_path,year,f),out_dir)
            continue

        logging.info('file: {}'.format(f))
        
        nc_obj = nc.Dataset(os.path.join(in_path, year, f))

        lon = np.asarray(nc_obj.variables['longitude'][:]) 
        
        # convert longitude from [0,360] to [-180,180]
        new_lon = lon - 180
       
        lat = np.asarray(nc_obj.variables['latitude'][:])
        
        time = np.asarray(nc_obj.variables['time'][:])

        depth = np.asarray(nc_obj.variables['depth'][:])
        
        temp = np.asarray(nc_obj.variables['temperature'][:])
        logging.info('temp: {}'.format(temp.shape))

        sal = np.asarray(nc_obj.variables['salinity'][:])
        logging.info('sal: {}'.format(sal.shape))

        # %%  Move the right side of longitude to the left side
        R_temp = temp[:,:,int(len(new_lon)/2):]
        
        L_temp = temp[:,:,:int(len(new_lon)/2)]

        new_temp = np.concatenate((R_temp,L_temp),axis=2)
        
        del R_temp, L_temp

        R_sal = sal[:,:,int(len(new_lon)/2):]
        
        L_sal = sal[:,:,:int(len(new_lon)/2)]

        new_sal = np.concatenate((R_sal, L_sal),axis=2)
        
        del R_sal, L_sal

        # %% 对比拼接前后
        plt.subplot(221)
        plt.imshow(temp[0], extent=(np.min(lon), np.max(lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('Original Temperature d5_mean')

        plt.subplot(222)
        plt.imshow(new_temp[0], extent=(np.min(new_lon), np.max(new_lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('new Temperature d5_mean')

        plt.subplot(223)
        plt.imshow(sal[0], extent=(np.min(lon), np.max(lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('Original Salinity d5_mean')

        plt.subplot(224)
        plt.imshow(new_sal[0], extent=(np.min(new_lon), np.max(new_lon), np.min(lat), np.max(lat)), origin='lower')
        plt.title('new Salinity d5_mean')
        plt.show()
        sys.exit(2)
        
        # %% save to a new nc
        logging.info('====== saving to {} ======'.format(out_file))
        out_nc = nc.Dataset(out_file, 'w', format='NETCDF4')

        out_nc.createDimension('latsize', len(lat))
        out_nc.createDimension('lonsize', len(new_lon))
        out_nc.createDimension('depth', len(depth))
        out_nc.createDimension('t', len(time))

        times = out_nc.createVariable('time', np.int32, ('t',))
        latitude = out_nc.createVariable('latitude', np.float32, ('latsize',))
        longitude = out_nc.createVariable('longitude', np.float32, ('lonsize',))
        dep = out_nc.createVariable('depth', np.float32, ('depth',))
        temp = out_nc.createVariable('temperature', np.float32, ('depth', 'latsize','lonsize'))
        sal = out_nc.createVariable('salinity', np.float32, ('depth', 'latsize','lonsize'))

        times.units = 'date'
        times.avg_peroid = str(len(time))+' day'
        latitude.units ='degree_north'
        longitude.units = 'degree_east'
        temp.units = 'Celsius degree'
        sal.units = 'psu'
        
        times[:] = time
        latitude[:] = lat
        longitude[:]= new_lon
        dep[:] = depth
        temp[:] = new_temp
        sal[:] = new_sal

        nc.Dataset.close(out_nc)
        
       # break



if __name__ == '__main__':
   
    dataset_path='/share_data/aigroup/DATA/01_In_situ_observations/NMDIS/IOCLevel'
    
    station_info_path ='/share_data/xyf/IOCLevel_Analysis/station_info.csv'    
    
    csv_save_dir ='/share_data/xyf/IOCLevel_Analysis/IOClevel_data'
    
    marked_save_path='/share_data/xyf/IOCLevel_Analysis/marked_map.svg'

    IOCLevel_dir = '/share_data/xyf/IOCLevel_Analysis/IOClevel_data/'
    
    monthly_IOClevel = '/share_data/xyf/IOCLevel_Analysis/monthly_IOClevel.csv'
    
    curve_pic_path = '/share_data/xyf/IOCLevel_Analysis/IOClevel_curve.svg'
    
    # 记录出现过的所有站点（合并重复出现的站点），记录站名、经度、纬度、出现次数
    # statistic_all_stations(dataset_path,station_info_path)

    # 统计每年每月服役的站点(服役的记1，未服役的记0)
    # statistic_stations_in_service(dataset_path, station_info_path)
    
    # 获取日期、站名、纬度、经度、连续缺失值的个数、24小时的逐时潮高,作为一行记录保存在csv中,csv按年划分（2015.csv,2016.csv,...）    
    # write_to_csv(dataset_path,csv_save_dir)

    # 在地图上标注出所有station的位置
    # mark_all_stations_on_map(station_info_path, marked_save_path)

    # 统计月平均潮高
    # monthly_mean_IOClevel(station_info_path, IOCLevel_dir, monthly_IOClevel)
    
    # 画出每个站点5年的月平均逐月潮高变化曲线
    # draw_monthly_mean_curve(monthly_IOClevel, curve_pic_path)

    # Process original CCMP data and interpolat to hycom, convert the longitude from 0~360 to -180~180
    # year = '2019'
    # ccmp_interp_to_hycom(year,hycom_file, ccmp_in_path, interp_ccmp_out_path)
    
    # Process original sla data and interpolat to hycom, convert the longitude from 0~360 to -180~180
    # year = '2019'
    # sla_interp_to_hycom(year, hycom_file, sla_in_path, interp_sla_out_path)
    
    # Process daily CCMP data and average the 5-day data per month(1~5, 6~10, 11~15, ....)
    # year = '2019'
    # ccmp_d5_mean(year, interp_ccmp_out_path, ccmp_d5_mean_path)

    # Process daily SLA data and average the 5-day data per month(1-5, 6-10, 11-15, ...)
    year = '2019'
    sla_d5_mean(year, interp_sla_out_path, sla_d5_mean_path)
    
    # Process daily SSS data and average the 5-day data per month(1-5, 6-10, 11-15, ...)
    # year = '2018'
    # sss_d5_mean(year, sss_out_path, sss_d5_mean_path)
    
    # Process daily SST data and average the 5-day data per month(1-5, 6-10, 11-15, ...)
    # year = '2019'
    # sst_d5_mean(year, sst_out_path, sst_d5_mean_path)

    # convert Hycom longitude range after 20170201: 0~360 ==> -180~180
    # y = '2019'
    # convert_hycom_longitude_range(y, hycom_out_path)
    
    # convert Hycom_d5_mean longitude range after 201702_1: 0~360 ==> -180~180
    # y = '2019'
    # convert_hycom_d5_mean_lon_range(y, hycom_d5_mean_path, new_hycom_d5_mean_path)

    # Process original sla data and interpolat to hycom, convert the longitude from 0~360 to -180~180
    # year = '2019'
    # sla_interp_to_hycom(year, hycom_file, sla_in_path, interp_sla_out_path)
