
"""
A series of more advanced functions. Incorporate with pandas dataframe, 
sunpy timeseries and more.

By Jay
"""

import pandas
import numpy as np
import sunpy.map

import os

from sunpy.time import TimeRange
import datetime

import json


def datalist_to_dataframe(data, column_name="intensity"):

    """Convert a datalist to dataframe."""
    times = []
    for timestr in data[0]:
        time = datetime.datetime.strptime(timestr, '%Y%m%dT%H%M%S')
        times.append(time)
    dataframe = pandas.DataFrame(data[1], index=times, columns=[column_name])

    return dataframe


def data_int(df_data, method='increase'):
    """
    Takes in a dataframe data and returns the integral of intensities.

    Offers 2 method of integration. Increase and all. Increase calculate only the
    increased energy.
    """

    print('Calculating integration...')
    sum = 0
    i0 = df_data.values[0][0]
    for j in range(len(df_data.index) - 1):
        dtime = TimeRange(df_data.index[j], df_data.index[j + 1]).seconds.value
        i1 = df_data.values[j][0]
        i2 = df_data.values[j + 1][0]

        if method == 'increase':
            sum += (i1 - i0) * dtime + abs(i2 - i1) * dtime / 2
        elif method == 'all':
            sum += i1 * dtime + abs(i2 - i1) * dtime / 2

    print('Integration ended successfully', sum)
    return sum

def write_json(data, data_name, file_name, file_dir):
    """
    Store data into a json file in file_path. Create one if there
    is no existing json file. Add an item to the dictionary if there is already a
    json file. For the dictionary in the json file: data_dict[data_name] = df_data.

    The data are supposed to be stored in json file as dicionary. For dataframe
    data we make a list of files, one event one file. For intgrated data we make
    one file out of all datas. So it should be only the one json file with integrated
    data that is passed to комиссар.
    """
    print('Writing data to a json file...')
    file_path = os.path.join(file_dir, file_name)
    if os.path.exists(file_path) is True:
        print('Writing to existing file:', file_path)
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        data_dict[data_name] = data
    else:
        print('Creating new file', file_path)
        data_dict = {}
        data_dict[data_name] = data
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)
    print('File written successfully.\n')

def mc_select(mapsequence, save_to=None, mapnum=None):

    """
    Take in a mapsequence, show the mapsequence and ask for an input.
    If the input is an float number between 0 and 1, return this number.
    If the input is an integer, save the corresponding map in the mapsequence and
    return nothing.

    Used to fine-tune and save contour cut image.
    """

    mapsequence.quicklook()
    if mapnum is None:
        print('\nPlease examine the mapsequence and choose the index of the disired contour (integer).')
        print('If none of the maps is good, input the new contour ratio (float between 0 and 1).')
        x = input()
        x = float(x)
        if x > 0 and x < 1: # 若没有合适的图像，返回一个新的contour ratio
            return x
        elif int(x) in range(len(mapsequence)):
            mapnum = int(x)

    print('Map at index', mapnum, 'is saved.')
    map = mapsequence[mapnum]
    # 存储fits
    if not save_to is None:
        map.save(save_to)

    # return mapnum

def get_peak_index(object, ban, type='mapsequence'):

    """
    Find the index of a map of which the max point is the biggest in the 
    mapsequence or fits list.

    Roughly used to locate the index of the peak in a mapsequence or fits list.
    """

    print('Finding peak index...')
    max = 0
    for i in range(len(object)):
        # Skip for problematic indices.
        if i in ban:
            continue

        if type == 'mapsequence':
            map = object[i]
        elif type == 'fitslist':
            map = sunpy.map.Map(object[i])
        
        map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi
        localmax = map.data.max()
        if localmax > max:
            max = localmax
            max_index = i
    print('Peak index found:', max_index, '\n')
    return max_index
