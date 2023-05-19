# -*- coding: utf-8 -*-
"""
@Features:
@Author: L.F. Zhan
@Date：2023/5/19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as colors
import cartopy.io.shapereader as shpreader
from scipy.interpolate import Rbf
from shapely.ops import unary_union
import shapefile
from shapely.geometry import Polygon
import cartopy.crs as ccrs
import datetime
import re

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文字符
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 创建气温色带
temperature_colors_warm = ['#D8F20C', '#FFDA05', '#E6A216', '#FC760D', '#F53F18']  # 气温暖色
cmaps_tem_p = colors.LinearSegmentedColormap.from_list('mycmap', temperature_colors_warm)  # 正气温色带
temperature_colors_cool = ['#80419D', '#069CEE', '#52CC8D', '#50CA4B', '#EFFF2D']  # 气温冷色
cmaps_tem_n = colors.LinearSegmentedColormap.from_list('mycmap', temperature_colors_cool)  # 负气温色带
temperature_colors = temperature_colors_cool + temperature_colors_warm  # 正负气温
cmaps_tem = colors.LinearSegmentedColormap.from_list('mycmap', temperature_colors)  # 正负气温生成色带
# 创建降水色带
# colorslist = ['dodgerblue', 'blue', 'green', 'chartreuse', 'yellow',
#               'red']  # 'darkturquoise','lightblue','royalblue' 降水色标
rain_levels = [0, 0.1, 10, 25, 50, 100, 250, 25000]
rain_colors = ['#FFFFFF', '#A6F28F', '#38A800', '#61B8FF', '#0000FF', '#FA00FA', '#730000', '#400000']
cmaps_pre = colors.LinearSegmentedColormap.from_list('mycmap', rain_colors)  # CMA标准降水量色带

rain_colors_p = ['#EFFF2D', '#50CA4B', '#52CC8D', '#069CEE', '#80419D']  # 降水正距平百分率颜色
cmaps_pre_p = colors.LinearSegmentedColormap.from_list('mycmap', rain_colors_p)  # 创建降水正距平百分率色带

rain_colors_n = temperature_colors_warm  # 降水负距平百分率颜色
cmaps_pre_n = colors.LinearSegmentedColormap.from_list('mycmap', rain_colors_n)  # 创建降水负距平百分率色带

rain_anomaly = ['#F53F18', '#FC760D', '#E6A216', '#FFDA05', '#D8F20C'] + rain_colors_p
cmaps_pre_anomaly = colors.LinearSegmentedColormap.from_list('mycmap', rain_anomaly)  # 创建降水距平百分率色带


def read_data(file_path):  # 格式转换（系统生成的titles与数据不匹配）
    df0 = pd.read_csv(file_path, sep=',', encoding='GB2312')
    df1 = pd.read_csv(file_path, sep=',', encoding='GB2312', header=1)
    title_name = list(df0.columns)
    title_name.extend(['Unnamed' + str(i) for i in range(df1.shape[1] - len(title_name))])
    df = pd.read_csv(file_path, sep=',', encoding='GB2312', header=0, names=title_name)
    df87 = df.iloc[0:87, :]
    return df, df87


def yn_increasing(data):
    '''
    判断图例level取整后是否为递增序列
    :param data:图例值
    :return:当返回为1，表示是递增序列；返回为0，表示非递增序列
    '''
    flage = 1
    for i in range(len(data) - 1):
        if data[i + 1] <= data[i]:
            flage = 0
            break
    return flage


def draw_color(z, arg, fn, str_num):
    '''
    :param z: 插值序列
    :param arg: 颜色  降水：rain;气温：t;日照：sun
    :param fn: 保存的文件名
    :return:
    '''
    global left, right, top
    path0 = 'JXshp/dishi.shp'
    file = shapefile.Reader(path0)
    rec = file.shapeRecords()
    polygon = list()
    for r in rec:
        polygon.append(Polygon(r.shape.points))
    poly = unary_union(polygon)  # 并集
    ext = list(poly.exterior.coords)  # 外部点
    codes = [Path.MOVETO] + [Path.LINETO] * (len(ext) - 1) + [Path.CLOSEPOLY]
    #    codes += [Path.CLOSEPOLY]
    ext.append(ext[0])  # 起始点
    path = Path(np.array(ext), codes)
    patch = PathPatch(path, facecolor='None')

    x, y = df_sta['经度'], df_sta['纬度']
    xi = np.arange(113, 118.5, 0.01)
    yi = np.arange(24, 31, 0.01)
    olon, olat = np.meshgrid(xi, yi)

    # Rbf空间插值
    func = Rbf(x, y, z, function='linear')
    oz = func(olon, olat)
    ax = plt.axes(projection=ccrs.PlateCarree())
    box = [113.4, 118.7, 24.1, 30.4]
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.add_patch(patch)
    shp = list(shpreader.Reader(path0).geometries())
    x_unit, y_unit = 117.45, 27.2
    if arg == 't':
        level = np.floor(np.linspace(np.floor(np.min(oz)), np.percentile(oz, 85), 6))  # 向下取整
        if np.min(oz) < 0 and np.max(oz) > 0:
            norm = colors.TwoSlopeNorm(vmin=np.min(oz), vcenter=0, vmax=np.max(oz))
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem, norm=norm, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem, norm=norm, extend='both')
        elif np.max(oz) < 0:
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem_n, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem_n, extend='both')
        else:
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem_p, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem_p, extend='both')
        # 添加单位标注
        plt.text(x_unit, y_unit, '℃', size=8, weight=2)
    elif arg == 't_anomaly':
        level = np.linspace(np.floor(np.min(oz)), np.percentile(oz, 85), 6)  # 向下取整
        if np.min(oz) < 0 and np.max(oz) > 0:
            norm = colors.TwoSlopeNorm(vmin=np.min(oz), vcenter=0, vmax=np.max(oz))
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem, norm=norm, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem, norm=norm, extend='both')
        elif np.max(oz) < 0:
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem_n, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem_n, extend='both')
        else:
            if yn_increasing(level) == 1:
                pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_tem_p, extend='both')
            else:
                pic = plt.contourf(olon, olat, oz, cmap=cmaps_tem_p, extend='both')
        # 添加单位标注
        plt.text(x_unit, y_unit, '℃', size=8, weight=2)
    elif arg == 'rain':
        level = np.floor(np.linspace(np.floor(np.min(oz)), np.percentile(oz, 99), 6))  # 向下取整
        if yn_increasing(level) == 1:
            pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_pre, extend='both')
        else:
            pic = plt.contourf(olon, olat, oz, cmap=cmaps_pre, extend='both')
        plt.text(x_unit, y_unit, 'mm', size=8, weight=2)
    elif arg == 'rain_anomaly':
        # level = np.floor(np.linspace(np.floor(np.min(oz)), np.percentile(oz, 99), 6))  # 向下取整
        # level = np.floor(np.linspace(np.percentile(oz, 5), np.percentile(oz, 99), 6))  # 向下取整
        level = [-80, -50, -20, 0, 20, 50, 80]
        if yn_increasing(level) == 1:
            pic = plt.contourf(olon, olat, oz, level, cmap=cmaps_pre_anomaly, extend='both')
        else:
            pic = plt.contourf(olon, olat, oz, cmap=cmaps_pre_anomaly, extend='both')
        plt.text(x_unit, y_unit, '%', size=8, weight=2)
    elif arg == 'sun':
        level = np.floor(np.linspace(np.floor(np.min(oz)), np.percentile(oz, 85), 6))  # 向下取整
        if yn_increasing(level) == 1:
            pic = plt.contourf(olon, olat, oz, level, cmap=plt.cm.hot_r, extend='both')
        else:
            pic = plt.contourf(olon, olat, oz, cmap=plt.cm.hot_r, extend='both')
        plt.text(x_unit, y_unit, 'h', size=8, weight=2)

    for collection in pic.collections:
        collection.set_clip_path(patch)  # 设置显示区域
    # 绘制站点
    # plt.scatter(x, y, marker='.', s=10, c='none', edgecolors='k', linewidths=0.2)

    # 添加显示站名、数值标签
    right = [58622, 58517, 57894, 58512, 57883, 58701, 58506, 57895, 57899, 59093, 58634, 58510, 58635, 58514, 58693,
             58602, 58615, 58608, 57995, 57789, 58712]
    left = [58601, 58606]
    top = [57891, 57994, 58907, 59102, 57792, 57896, 58623, 57698, 58626, 58529, 58625, 58600, 58502,
           58519, 58814, 57694, 58509, 58713, 59092, 58612, 57696, 57991, 58903]
    font_size = 3
    for i in range(len(z)):
        plt.scatter(x[i], y[i], marker='.', s=10, c='none', edgecolors='k', linewidths=0.2)
        if df_sta.iloc[i, 1] in left:
            plt.text(x[i] - 0.43, y[i] - 0.02, str(df_sta['站名'][i]) + str(z[i]), size=font_size)
        elif df_sta.iloc[i, 1] in top:
            plt.text(x[i] - 0.2, y[i] + 0.05, str(df_sta['站名'][i]) + str(z[i]), size=font_size)
        elif df_sta.iloc[i, 1] in right:
            plt.text(x[i] + 0.05, y[i] - 0.02, str(df_sta['站名'][i]) + str(z[i]), size=font_size)
        else:  # 下方
            plt.text(x[i] - 0.2, y[i] - 0.11, str(df_sta['站名'][i]) + str(z[i]), size=font_size)

    # 添加地市边界
    ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor='black',
                      facecolor='none', alpha=0.3, linewidth=0.5)  # 加底图
    # 添加标题
    ele = re.findall('tem|tmax|tmin|rain|sun', fn)[0]
    if ele == 'tem' and arg == 't':
        title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均气温' + str_num + '℃'
    elif ele == 'tem' and arg == 't_anomaly':
        title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均气温距平' + str_num + '℃'
    elif ele == 'tmax':
        title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最高气温' + str_num + '℃'
    elif ele == 'tmin':
        title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最低气温' + str_num + '℃'
    elif ele == 'rain':
        if arg == 'rain_anomaly':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '降水距平百分率' + str_num + '%'
        else:
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '降水量' + str_num + 'mm'
    elif ele == 'sun':
        title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均日照时数' + str_num + 'h'

    text_make = '江西省气候中心' + datetime.datetime.now().strftime('%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日') + '制作'
    plt.title(title, size=6, y=0.95)
    plt.text(116.5, 24.5, text_make, size=5)

    # 保存图片
    fig = plt.gcf()
    fig.set_size_inches(6, 4)  # 设置图片大小
    plt.axis('off')  # 去除四边框框
    position = fig.add_axes([0.62, 0.2, 0.03, 0.28])  # 位置
    cb = plt.colorbar(pic, cax=position, orientation='vertical', format='%.1f', extendfrac='auto')  # 图例
    cb.ax.tick_params(labelsize=8, )

    fn_path = re.findall('tem|tmax|tmin|rain|sun', fn)[0]
    if arg == 'rain_anomaly':
        plt.savefig(r'C:\qhzxdata\\' + fn_path + '\\' + fn[:-4] + '-2.png', dpi=500, bbox_inches='tight')
    elif arg == 't_anomaly':
        plt.savefig(r'C:\qhzxdata\\' + fn_path + '\\' + fn[:-4] + '-2.png', dpi=500, bbox_inches='tight')
    else:
        plt.savefig(r'C:\qhzxdata\\' + fn_path + '\\' + fn[:-4] + '-1.png', dpi=500, bbox_inches='tight')
    plt.close()
    # plt.show()


def draw_rank(fn, rank):
    '''
    绘制排位图
    :return:
    '''
    path0 = 'JXshp/dishi.shp'
    file = shapefile.Reader(path0)
    rec = file.shapeRecords()
    polygon = list()
    for r in rec:
        polygon.append(Polygon(r.shape.points))
    poly = unary_union(polygon)  # 并集
    ext = list(poly.exterior.coords)  # 外部点
    codes = [Path.MOVETO] + [Path.LINETO] * (len(ext) - 1) + [Path.CLOSEPOLY]
    #    codes += [Path.CLOSEPOLY]
    ext.append(ext[0])  # 起始点
    path = Path(np.array(ext), codes)
    patch = PathPatch(path, facecolor='None')
    ax = plt.axes(projection=ccrs.PlateCarree())
    box = [113.4, 118.7, 24.1, 30.4]
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.add_patch(patch)
    shp = list(shpreader.Reader(path0).geometries())

    # 添加地市边界
    ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor='black',
                      facecolor='none', alpha=0.3, linewidth=0.5)  # 加底图

    # 添加title
    ele = re.findall('tem|tmax|tmin|rain|sun', fn)[0]
    if int(rank) <= (datetime.datetime.now().year - 1960) / 2:  # 判断rank是否是前半位还是后半位
        if ele == 'tem':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均气温排历史同期第' + rank + '高位'
        elif ele == 'tmax':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最高气温排历史同期第' + rank + '高位'
        elif ele == 'tmin':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最低气温排历史同期第' + rank + '高位'
        elif ele == 'rain':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '降水量排历史同期第' + rank + '高位'
        elif ele == 'sun':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均日照时数排历史同期第' + rank + '高位'
    else:
        rank_de = (datetime.datetime.now().year - 1960) - int(rank) + 1
        if ele == 'tem':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均气温排历史同期第' + str(rank_de) + '低位'
        elif ele == 'tmax':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最高气温排历史同期第' + str(rank_de) + '低位'
        elif ele == 'tmin':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均最低气温排历史同期第' + str(rank_de) + '低位'
        elif ele == 'rain':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '降水量排历史同期第' + str(rank_de) + '低位'
        elif ele == 'sun':
            title = re.findall(('\d.*-\d*'), fn)[0] + '江西省' + '平均日照时数排历史同期第' + str(rank_de) + '低位'
    plt.title(title, size=6, y=0.95)
    plt.text(116.5, 25, '注：' + '\n' * 2 + '1.标红为排位前五或后五位站点' + '\n' * 2 +
             '2.排序方法为1961年至今历史' + '\n' * 2 + '  同期数值由大到小排序', size=5)
    # 标注制作单位
    text_make = '江西省气候中心' + datetime.datetime.now().strftime('%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日') + '制作'
    plt.text(116.5, 24.5, text_make, size=5)

    # 添加显示站名、排位标签
    x, y = df_sta['经度'], df_sta['纬度']
    font_size = 3
    for i in range(len(df_sta)):
        plt.scatter(x[i], y[i], marker='.', s=10, c='none', edgecolors='k', linewidths=0.2)
        if df_sta.iloc[i, 1] in left:
            if df_sta['历史排位'][i] <= 5 or df_sta['历史排位'][i] >= datetime.datetime.now().year - 1964:
                plt.text(x[i] - 0.44, y[i] - 0.02, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), color='r',
                         size=font_size)
            else:
                plt.text(x[i] - 0.44, y[i] - 0.02, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), size=font_size)
        elif df_sta.iloc[i, 1] in top:
            if df_sta['历史排位'][i] <= 5 or df_sta['历史排位'][i] >= datetime.datetime.now().year - 1964:
                plt.text(x[i] - 0.2, y[i] + 0.05, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), color='r',
                         size=font_size)
            else:
                plt.text(x[i] - 0.2, y[i] + 0.05, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), size=font_size)
        elif df_sta.iloc[i, 1] in right:
            if df_sta['历史排位'][i] <= 5 or df_sta['历史排位'][i] >= datetime.datetime.now().year - 1964:
                plt.text(x[i] + 0.05, y[i] - 0.02, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), color='r',
                         size=font_size)
            else:
                plt.text(x[i] + 0.05, y[i] - 0.02, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), size=font_size)
        else:  # 下方
            if df_sta['历史排位'][i] <= 5 or df_sta['历史排位'][i] >= datetime.datetime.now().year - 1964:
                plt.text(x[i] - 0.15, y[i] - 0.12, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), color='r',
                         size=font_size)
            else:
                plt.text(x[i] - 0.15, y[i] - 0.12, str(df_sta['站名'][i]) + str(df_sta['历史排位'][i]), size=font_size)
    # 保存图片
    fig = plt.gcf()
    fig.set_size_inches(6, 4)  # 设置图片大小
    plt.axis('off')  # 去除四边框框
    fn_path = re.findall('tem|tmax|tmin|rain|sun', fn)[0]
    plt.savefig(r'C:\qhzxdata\\' + fn_path + '\\' + fn[:-4] + '-3.png', dpi=500, bbox_inches='tight')
    plt.close()


def generate_fn():
    '''
    生成当前日期前一天的文件名
    :return:
    '''
    global month
    yesday = datetime.datetime.now() + datetime.timedelta(days=-1)
    yesday_str = yesday.strftime('%Y%m%d')
    year = yesday.year
    month = yesday.month
    fn_thismonth_yesday, fn_01_yesday, fn_04_yesday, fn_07_yesday = [], [], [], []
    elements = ['tem-', 'tmax-', 'tmin-', 'rain-', 'sun-']
    if month >= 1 and month < 4:  # 只画1月以来及本月以来
        for element in elements:
            if element == 'rain-':
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '_20.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '_20.txt')
            else:
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '.txt')
    elif month >= 4 and month < 7:  # 画1月以来、4月以来及本月以来
        for element in elements:
            if element == 'rain-':
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '_20.txt')
                fn_04_yesday.append(element + str(year) + '0401-' + yesday_str + '_20.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '_20.txt')
            else:
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '.txt')
                fn_04_yesday.append(element + str(year) + '0401-' + yesday_str + '.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '.txt')
    elif month >= 7:  # 画1月以来、4月以来、7月以来及本月以来
        for element in elements:
            if element == 'rain-':
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '_20.txt')
                fn_04_yesday.append(element + str(year) + '0401-' + yesday_str + '_20.txt')
                fn_07_yesday.append(element + str(year) + '0701-' + yesday_str + '_20.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '_20.txt')
            else:
                fn_01_yesday.append(element + str(year) + '0101-' + yesday_str + '.txt')
                fn_04_yesday.append(element + str(year) + '0401-' + yesday_str + '.txt')
                fn_07_yesday.append(element + str(year) + '0701-' + yesday_str + '.txt')
                fn_thismonth_yesday.append(element + yesday_str[0:6] + '01-' + yesday_str + '.txt')
    return fn_thismonth_yesday, fn_01_yesday, fn_04_yesday, fn_07_yesday


if __name__ == '__main__':
    fn_thismon_ystd, fn_Jan_ystd, fn_Apr_ystd, fn_Jul_ystd = generate_fn()  # 生成对应时段文件名

    # 平均气温
    df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_thismon_ystd[0])
    draw_color(df_sta.iloc[:, 4], 't', fn_thismon_ystd[0], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_thismon_ystd[0], str(df['历史排位'][87]))
    df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Jan_ystd[0])
    draw_color(df_sta.iloc[:, 4], 't', fn_Jan_ystd[0], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_Jan_ystd[0], str(df['历史排位'][87]))
    # 平均气温距平
    df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_thismon_ystd[0])
    draw_color(df_sta.iloc[:, 5], 't_anomaly', fn_thismon_ystd[0], str(round(df.iloc[87, 5], 1)))
    df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Jan_ystd[0])
    draw_color(df_sta.iloc[:, 5], 't_anomaly', fn_Jan_ystd[0], str(round(df.iloc[87, 5], 1)))
    # 最高气温
    df, df_sta = read_data(r'C:\qhzxdata\tmax' + '\\' + fn_thismon_ystd[1])
    draw_color(df_sta.iloc[:, 4], 't', fn_thismon_ystd[1], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_thismon_ystd[1], str(df['历史排位'][87]))
    df, df_sta = read_data(r'C:\qhzxdata\tmax' + '\\' + fn_Jan_ystd[1])
    draw_color(df_sta.iloc[:, 4], 't', fn_Jan_ystd[1], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_Jan_ystd[1], str(df['历史排位'][87]))
    # 最低气温
    df, df_sta = read_data(r'C:\qhzxdata\tmin' + '\\' + fn_thismon_ystd[2])
    draw_color(df_sta.iloc[:, 4], 't', fn_thismon_ystd[2], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_thismon_ystd[2], str(df['历史排位'][87]))
    df, df_sta = read_data(r'C:\qhzxdata\tmin' + '\\' + fn_Jan_ystd[2])
    draw_color(df_sta.iloc[:, 4], 't', fn_Jan_ystd[2], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_Jan_ystd[2], str(df['历史排位'][87]))
    # 降水
    df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_thismon_ystd[3])
    draw_color(df_sta.iloc[:, 4], 'rain', fn_thismon_ystd[3], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_thismon_ystd[3], str(df['历史排位'][87]))
    df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Jan_ystd[3])
    draw_color(df_sta.iloc[:, 4], 'rain', fn_Jan_ystd[3], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_Jan_ystd[3], str(df['历史排位'][87]))
    # 降水距平
    df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_thismon_ystd[3])
    draw_color(df_sta.iloc[:, 5], 'rain_anomaly', fn_thismon_ystd[3], str(round(df.iloc[87, 5], 1)))
    df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Jan_ystd[3])
    draw_color(df_sta.iloc[:, 5], 'rain_anomaly', fn_Jan_ystd[3], str(round(df.iloc[87, 5], 1)))
    # 日照
    df, df_sta = read_data(r'C:\qhzxdata\sun' + '\\' + fn_thismon_ystd[4])
    draw_color(df_sta.iloc[:, 4], 'sun', fn_thismon_ystd[4], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_thismon_ystd[4], str(df['历史排位'][87]))
    df, df_sta = read_data(r'C:\qhzxdata\sun' + '\\' + fn_Jan_ystd[4])
    draw_color(df_sta.iloc[:, 4], 'sun', fn_Jan_ystd[4], str(round(df.iloc[87, 4], 1)))
    draw_rank(fn_Jan_ystd[4], str(df['历史排位'][87]))
    if month >= 4:  # Since April
        # 平均气温
        df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Apr_ystd[0])
        draw_color(df_sta.iloc[:, 4], 't', fn_Apr_ystd[0], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Apr_ystd[0], str(df['历史排位'][87]))
        # 平均气温距平
        df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Apr_ystd[0])
        draw_color(df_sta.iloc[:, 5], 't_anomaly', fn_Apr_ystd[0], str(round(df.iloc[87, 5], 1)))
        # 最高气温
        df, df_sta = read_data(r'C:\qhzxdata\tmax' + '\\' + fn_Apr_ystd[1])
        draw_color(df_sta.iloc[:, 4], 't', fn_Apr_ystd[1], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Apr_ystd[1], str(df['历史排位'][87]))
        # 最低气温
        df, df_sta = read_data(r'C:\qhzxdata\tmin' + '\\' + fn_Apr_ystd[2])
        draw_color(df_sta.iloc[:, 4], 't', fn_Apr_ystd[2], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Apr_ystd[2], str(df['历史排位'][87]))
        # 降水
        df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Apr_ystd[3])
        draw_color(df_sta.iloc[:, 4], 'rain', fn_Apr_ystd[3], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Apr_ystd[3], str(df['历史排位'][87]))
        # 降水距平
        df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Apr_ystd[3])
        draw_color(df_sta.iloc[:, 5], 'rain_anomaly', fn_Apr_ystd[3], str(round(df.iloc[87, 5], 1)))
        # 日照
        df, df_sta = read_data(r'C:\qhzxdata\sun' + '\\' + fn_Apr_ystd[4])
        draw_color(df_sta.iloc[:, 4], 'sun', fn_Apr_ystd[4], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Apr_ystd[4], str(df['历史排位'][87]))
    if month >= 7:  # Since July
        # 平均气温
        df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Jul_ystd[0])
        draw_color(df_sta.iloc[:, 4], 't', fn_Jul_ystd[0], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Jul_ystd[0], str(df['历史排位'][87]))
        # 气温距平
        df, df_sta = read_data(r'C:\qhzxdata\tem' + '\\' + fn_Jul_ystd[0])
        draw_color(df_sta.iloc[:, 5], 't_anomaly', fn_Jul_ystd[0], str(round(df.iloc[87, 5], 1)))
        # 最高气温
        df, df_sta = read_data(r'C:\qhzxdata\tmax' + '\\' + fn_Jul_ystd[1])
        draw_color(df_sta.iloc[:, 4], 't', fn_Jul_ystd[1], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Jul_ystd[1], str(df['历史排位'][87]))
        # 最低气温
        df, df_sta = read_data(r'C:\qhzxdata\tmin' + '\\' + fn_Jul_ystd[2])
        draw_color(df_sta.iloc[:, 4], 't', fn_Jul_ystd[2], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Jul_ystd[2], str(df['历史排位'][87]))
        # 降水
        df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Jul_ystd[3])
        draw_color(df_sta.iloc[:, 4], 'rain', fn_Jul_ystd[3], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Jul_ystd[3], str(df['历史排位'][87]))
        # 降水距平
        df, df_sta = read_data(r'C:\qhzxdata\rain' + '\\' + fn_Jul_ystd[3])
        draw_color(df_sta.iloc[:, 5], 'rain_anomaly', fn_Jul_ystd[3], str(round(df.iloc[87, 5], 1)))
        # 日照
        df, df_sta = read_data(r'C:\qhzxdata\sun' + '\\' + fn_Jul_ystd[4])
        draw_color(df_sta.iloc[:, 4], 'sun', fn_Jul_ystd[4], str(round(df.iloc[87, 4], 1)))
        draw_rank(fn_Jul_ystd[4], str(df['历史排位'][87]))
