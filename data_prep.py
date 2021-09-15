# A series of function to prepare data for GAN
import flares as fl
import sunpy.map
import numpy as np
import os
import tensorflow as tf
import astropy.units as u

from sunpy.time import parse_time as ParseTime
from astropy.io import fits as fts


def RemoveOverflowImage(fits_list):
    """Remove image with duplicate max points, i.e. with overflow pixels."""
    print('Removing overflow files...')
    count = 0
    for fits in fits_list:
        if OverflowJudge(fits):
            fits_list.remove(fits)
            count += 1
    print(count, "overflow files removed.")
    return fits_list

def OverflowJudge(fits, type='value'):
    """A function to tell if the given fits data have overflow points or not by
    checking the number of max points."""
    map = sunpy.map.Map(fits)

    if type == 'max':
        pixel_pos = np.argwhere(map.data == map.data.max())
        criteria = 1
    elif type == 'value':
        pixel_pos = np.argwhere(map.data > 16000) # 标准针对aia euv估计的
        criteria = 10

    if len(pixel_pos) > criteria:
        flag = True
    else:
        flag = False

    return flag

def GetMultiChannelData(event_dir, wavelengths, target_wave = '1216A'):
    """Get a time aligned fits list with multiple channel. HMI continuum being
    the target channel. Need to change a few things if you want other channel to
    be target channel. 
    The output list being the following manner:
    For input wavelengths = [94A, 131A, Whitelight]
    [[fits_94A, fits_131A, fits_Whitelight]
    [fits_94A, fits_131A, fits_Whitelight]]"""
    fits_dict = {}
    for wave in wavelengths:
        fits_list = fl.get_fits_list(os.path.join(event_dir, wave))
        # fits_list = RemoveOverflowImage(fits_list) # You can annotate this line to allow overflow image pass through.
        fits_dict[wave] = fits_list # Create a dictionary to save those fits_list(s)
    # Align time to make multiple channel happen
    dataset_fits_list = []
    for fits_target in fits_dict[target_wave]:
        t1 = TimeFits(fits_target)
        temp = []
        flag = True
        # Following loop is to construct a temp. For example:
        # temp = [fits_94A, fits_131A, fits_Whitelight]
        for wave in wavelengths:
            if wave == target_wave:
                continue
            for fits_aia in fits_dict[wave]:
                t2 = TimeFits(fits_aia)
                dt = Sec(t2 - t1)
                if dt < -45: # t2比t1早30秒以上，为了提高遍历速度去掉它
                    fits_dict[wave].remove(fits_aia)
                elif abs(dt) < 45: # 时间间隔小于30秒
                    # print(wave, 'Time matched:', t1, t2, dt)
                    temp.append(fits_aia)
                    break
                elif dt > 45: # t2比t1晚30秒以上，说明在t2的遍历中没找到和t1的匹配
                    flag = False
                    break
            if flag is False:
                # print('Data unmatched.')
                break
        if len(temp) != len(wavelengths) - 1:
            flag = False
        if flag is True:
            temp.append(fits_target)
            print('Data matched:', len(temp), 'channels')
            dataset_fits_list.append(temp)
            for fits_del, wave_del in zip(temp, wavelengths):
                fits_dict[wave_del].remove(fits_del)
                # 从列表中删除已经存下的fits，避免重复匹配
    return dataset_fits_list

def Sec(day):
    return day * 24 * 3600

def TimeFits(fits):
    """A quick way to read the time of a file. Using astropy to read only the
    headers."""
    # 读取头文件，其中vault和solo头文件的hdu number为0
    try:
        header = fts.getheader(fits, 1)
    except IndexError:
        header = fts.getheader(fits, 0)
    time = ParseTime(header['DATE-OBS'])
    return time

def TelescopeFits(fits):
    """A quick way to read the instrument used. Using astropy to read only the
    headers.
    aia: TELESCOP='SDO/AIA'
    hmi: TELESCOP='SDO/HMI'
    eui hri: TELESCOP='SOLO/EUI/HRI_EUV'
    eui fsi: TELESCOP='SOLO/EUI/FSI'
    vault(the rocket): ORIGIN='VAULT2.0 Rocket 1216 A' (have no 'TELESCOP' label)
    """
    try:
        header = fts.getheader(fits, 1)
    except IndexError:
        header = fts.getheader(fits, 0)
    telescope = header['TELESCOP']
    return telescope


def SaveDSFitsList(dataset_fits_list, cut_radius, out_radius, output_dir, type='lya'):
    """Save data_set_fits_list from GetMultiChannelData. Data is stored as fits 
    files in paths according to its channel:
    output_dir//0//xxxx.fits
    output_dir//1//xxxx.fits
    output_dir//2//xxxx.fits
    
    Should work on vault as well.
    
    type = wl: cut around max point, fit for flares.
    type = lya: cut around center point, fit for lya images."""
    channels = len(dataset_fits_list[0])
    paths = []
    # Create output directory for multiple channels
    for i in range(channels):
        path = os.path.join(output_dir, str(i))
        if os.path.exists(path) is False:
            os.makedirs(path)
        paths.append(path)
    # 开始裁剪和存储
    for set in dataset_fits_list:
        # 全日面的处理，只做resample
        if type == 'full_disk':
            for i in range(channels):
                map = sunpy.map.Map(set[i])
                fl.square_cut(map, None, None, [out_radius*2, out_radius*2], paths[i])

        else:
            proceed = False # To avoid the bug of max point outside of solar disk
            while not proceed is True:
                temp_files = []
                j = 0
                try:
                    # 根据type选项找剪裁中心点
                    if type == 'whitelight':
                        coord = fl.find_max_coord(set[j])
                    elif type == 'lya':
                        coord = fl.find_center_coord(set[-1])
                    bottom_left, top_right = fl.get_diag_coord(sunpy.map.Map(set[j]),
                    coord, cut_radius)
                    # 剪裁并储存文件，同时先记录文件名
                    for i in range(channels):
                        map = sunpy.map.Map(set[i])
                        if type == 'lya' and i != 9:
                            map = map.rotate(angle=18 * u.deg) # 旋转aia图像以对正
                        temp_file = fl.square_cut(map, bottom_left,
                        top_right, [out_radius*2, out_radius*2], paths[i])
                        temp_files.append(temp_file)
                # 若出错，删除这些文件
                except ValueError:
                    print('ValueError captured')
                    j += 1
                    for temp_file in temp_files:
                        os.remove(temp_file)
                # 若没有出错，继续即可
                else:
                    proceed = True

def OverflowDSRemove(dataset_fits_list):
    for set in dataset_fits_list:
        for fits in set:
            if OverflowJudge(fits) is True:
                dataset_fits_list.remove(set)
                break
    return dataset_fits_list
        
def AIAPrep(fits, Scaler, Cropper):
    map = sunpy.map.Map(fits)
    data = map.data
    exptime = map.exposure_time.value
    data = data / exptime
    data = Scaler(data)
    data = tf.expand_dims(data, 2)
    data = tf.expand_dims(data, 0)
    data = Cropper(data)
    return data[0]

def HMIPrep(fits, Scaler, Cropper):
    map = sunpy.map.Map(fits)
    data = map.data
    data = np.flip(data, axis=0)
    data = np.flip(data, axis=1)
    data[np.where(np.isnan(data))] = 0
    # substitute 'nan' with 0 for hmi
    data = Scaler(data)
    data = tf.expand_dims(data, 2)
    data = tf.expand_dims(data, 0)
    data = Cropper(data)
    return data[0]

def VAULTPrep(fits, Scaler, Cropper):
    map = sunpy.map.Map(fits)
    data = map.data
    data = Scaler(data)
    data = tf.expand_dims(data, 2)
    data = tf.expand_dims(data, 0)
    data = Cropper(data)
    return data[0]