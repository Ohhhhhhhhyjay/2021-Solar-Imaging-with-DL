
"""
A series of basic functions that operate directly on maps.
By Jay
latest version: v0.7 @2021.3.20
"""

import os
import numpy as np

import sunpy.map

import astropy.units as u
# from astropy.coordinates import SkyCoord
# from sunpy.coordinates import frames
from sunpy.coordinates import RotatedSunFrame

from scipy import ndimage

import datetime
import csv

from sunpy.image.coalignment import mapsequence_coalign_by_match_template as mc_coalign
from sunpy.physics.solar_rotation import mapsequence_solar_derotate as mc_derotate

# Global variables are capitalized.
DOWNLOAD_PATH = r'E:\Researches\2020EUVSolarFlare\Data_download'
ROOT_PATH = r'E:\Researches\2020EUVSolarFlare\Data'
OUTPUT_PATH = r'E:\Researches\2020EUVSolarFlare\Data_output'
WAVELENGTHS = ((94, 131, 171, 193, 211, 304, 335, 1600, 1700, 6173.0),\
    ('94A', '131A', '171A', '193A', '211A', '304A', '335A', '1600A', '1700A', \
        'Whitelight'))

WAVELENGTHS_LYA = ((94, 131, 171, 193, 211, 304, 335, 1600, 1700, 1216),\
    ('94A', '131A', '171A', '193A', '211A', '304A', '335A', '1600A', '1700A', \
        '1216A'))


def get_data_dir(root_path=ROOT_PATH, wavelengths=WAVELENGTHS):

    """
    Get the data from the root_path organized in a specific manner: 
    root_path\\eventYYYYMMDDThh\\???A(or "Whitelight" in the case of 6173.0 
    angstrom). 

    Returns a 2-dimensional dictionary of directories. data_dir[event] 
    is the list of event and data_dir[event][wavelength] is the path
    of a certain event and wavelength. Under the directory of 
    data_dir[event][wavelength] is a bunch of fits files, which is accessible 
    through os.walk() or get_fits_list() (2021.4.7)
    """

    print('Retriving data directory...')
    data_dir = {} # Use dictionary!!!
    for root, dirs, files in os.walk(root_path):
        for i in range(len(dirs)):
            dirname = dirs[i]
            if dirname[:5] == 'event':
                data_dir[dirname] = {}
                for wavename in wavelengths[1]:
                    data_dir[dirname][wavename] = os.path.join(root, dirname, wavename)
    
    print('Data directory dict retrieved.')
    # print(data_dir)
    return data_dir


def get_fits_list(data_dir, keyword=None):

    """A simple function takes in a directory and return the path of the fits
    files within. 因为懒得写一百遍os.walk()."""

    # print('Retrieving fits list...')
    fits_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file[-4:] == 'fits' and (keyword is None or keyword in file):
                fits_list.append(os.path.join(root, file))
    # print(fits_list)
    # print('Fits list retrieved.')
    return fits_list

    

def find_max_coord(fits):

    """Takes a fits file and return its max coordinate, works well for both aia 
    and hmi continuum (whitelight). If there is more than one max point (overflow), 
    calculate and return the mean position of them."""

    # print('Finding max coordinates...')
    map = sunpy.map.Map(fits)

    map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi

    # print('Max value:', map.data.max())
    pixel_pos = np.argwhere(map.data == map.data.max())
    pixel_pos_mean = np.mean(pixel_pos, axis=0) * u.pixel
    # 这里xy坐标是反过来的，前面pixel pos的xy就是反的，不知道为什么。暂时这样解决吧。
    max_coord = map.pixel_to_world(pixel_pos_mean[1], pixel_pos_mean[0])
    
    # print('Max coordinates found...')
    print('Max pixel on aia map:', pixel_pos_mean)
    # print('Max coordinate:', max_coord)
    return max_coord

def find_center_coord(fits):

    """Takes a fits file and return its center coordinate.
    Works for vault lya image, hmi, and aia image."""

    map = sunpy.map.Map(fits)

    arcsecs = u.def_unit('arcsecs', 1 * u.arcsec) # 对于vault数据，要获取坐标需要
    u.add_enabled_units([arcsecs])
    map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi

    x_len = len(map.data[0])
    y_len = len(map.data)

    coord = map.pixel_to_world(x_len / 2 * u.pixel, y_len / 2 * u.pixel)
    return coord

def rotate_coord(coord, fits_list):

    """Rotate the given point (using SkyCoord) on one of the fits file (not 
    necessarily be the first one since RotatedSunFrame is able to take negative 
    rotate durations) of a series of fits, the path of which is stored in 
    fits_list. Return a series of rotated coordinates, each corresponds to the 
    fits in fits_list."""

    print('Rotating coordinates...')
    times = []
    for fits in fits_list:
        map = sunpy.map.Map(fits)
        times.append(map.date)
    rotated_coord = RotatedSunFrame(base=coord, rotated_time=times)
    rotated_coords = rotated_coord.transform_to(coord)
    print('Rotated coordinates:', rotated_coords)
    return rotated_coords



def contour_cut(map, prefilter=None, ratio=0.80, sigma=14, save_to=None):

    """
    A function that select and cut the brightest region of a map smoothly. Sigma
    is a parameter used by gaussian filter to smooth out the edge.
    
    We choose the default criterion that the data should be at least 50% of the 
    maximum value. Pixels with intensity values greater than this are included 
    in the mask, while all other pixels are excluded.
    
    Prefilter is saved filter from existing operation. A sunpy map.
    Note that if prefilter is not None, ratio and sigma do not work. There would
    be no gaussian filter calculation involves ratio and sigma then. (2021.4.7)
    """

    map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi

    if prefilter is None:
        mask = map.data < map.max() * ratio
        filtered = ndimage.gaussian_filter(map.data * ~mask, sigma) # smoothout the area
        filtered[filtered < 0.01 * map.mean()] = 0 # 取小于平均值百分之一的数据为零。
    else:
        filtered = prefilter.data

    data_cut = filtered # 应该是datacut要初始化，我忘了为啥加这句话了
    data_cut = np.where(filtered != 0, map.data, 0)
    map_cut = sunpy.map.Map(data_cut, map.meta)
    labels, n = ndimage.label(map_cut.data)
    print('||', n, 'REGION(s) DETECTED.||' )

    if not save_to is None:
        time = sunpydate_to_str(map_cut.date)
        file_save = os.path.join(save_to, 'contour_cut' + time + '.fits')
        map_cut.save(file_save)

    # map_cut.peek() # check the result map_cut if you want.
    return map_cut

def mc_contour_cut(mapsequence, prefilter=None, ratio=0.80, sigma=14, save_to=None):

    """
    A function that incorporates contour_cut and mapsequence to operate on
    mapsequence. Returns a cut mapsequence ready for later operation.
    """
    print('Calculating contours of images...')
    maps = []
    for map in mapsequence:
        map_cut = contour_cut(map, prefilter, ratio, sigma, save_to)
        maps.append(map_cut)
    mapsequence_cut = sunpy.map.Map(maps, sequence=True)

    print('Mapsequence contour cut ended successfully.')
    return mapsequence_cut

def get_diag_coord(map, coord, radius):

    arcsecs = u.def_unit('arcsecs', 1 * u.arcsec) # 对于vault数据，要获取坐标需要
    u.add_enabled_units([arcsecs])
    map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi

    pix = map.world_to_pixel(coord)
    x = pix[0].value
    y = pix[1].value
    pix = [int(x) * u.pixel, int(y) * u.pixel]
    # print('Corresponding pixel on hmi map:', pix)
    radius = radius * u.pixel
    bottom_left = map.pixel_to_world(pix[0] - radius,\
        pix[1] - radius)
    top_right = map.pixel_to_world(pix[0] + radius,\
        pix[1] + radius)
    
    return bottom_left, top_right

def square_cut(map, bottom_left, top_right, dimensions=None, save_to=None):

    """A function that cut out a given full map to a square region with a given 
    radius, i.e. half of the side length. Returns the map cut out. One can pass 
    a path and filename to save the cut out map as fits.

    dimensions = [xxx, xxx], no need for units.
    """

    arcsecs = u.def_unit('arcsecs', 1 * u.arcsec) # 对于vault数据，要获取坐标需要
    u.add_enabled_units([arcsecs])
    map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi
    
    # 裁剪并修改大小
    if not bottom_left is None and not top_right is None:
        map = map.submap(bottom_left, top_right)
    if not dimensions is None:
        map = TestRecToSquare(map)
        map = map.resample(dimensions * u.pixel)

    if not save_to is None:
        time = sunpydate_to_str(map.date)
        file_save = os.path.join(save_to, 
            map.detector + str(map.wavelength.value) + 'square_cut' + time + '.fits')
        # print(os.path.exists(file_save))
        if os.path.exists(file_save) is False:
            map.save(file_save)

        return file_save

    # map_cut.peek()
    elif save_to is None:
        return map

def TestRecToSquare(map):
    """
    A not-so-general function. Cutting left margin off a map.
    """
    shape = map.data.shape
    if shape[1] > shape[0]:
        bl = [shape[1]-shape[0], 0] * u.pixel
        tr = [shape[1], shape[0]] * u.pixel
        map = map.submap(bottom_left=bl, top_right=tr)
    return map

def mc_square_cut(fits_list, coord, radius):

    """
    A function that incorporates square_cut and mapsequence to operate on
    multiple maps. Returns a cut mapsequence ready for later operation. It takes 
    a fits_list for input because the complete data are too large to convert to 
    mapsequence  directly.
    """

    print('Loading mapsequence square cut...')
    maps = []
    for fits in fits_list:
        map = sunpy.map.Map(fits)
        map_cut = square_cut(map, coord, radius)
        maps.append(map_cut)
    mapsequence_cut = sunpy.map.Map(maps, sequence=True)
    print('Mapsequence square cut ended successfully.')
    return mapsequence_cut

def maps_difference(map_begin, map_now, save_to=None):
    
    """
    Returns the map of map_now - map_begin, with metadata (headers and all) 
    of map_now. Usually map_now is the later data. This way we can show the 
    change of intensity over time.
    """

    data_diff = map_now.data - map_begin.data
    map_diff = sunpy.map.Map(data_diff, map_now.meta)

    if not save_to is None:
        time = sunpydate_to_str(map_diff.date)
        file_save = os.path.join(save_to, 'difference' + time + '.fits')
        map_diff.save(file_save)

    # map_diff.peek()
    return map_diff

# Suggested
def mc_align(mapsequence, keyword='coalign'):

    """
    INPUT: a mapsequence, a keyword:'derotate'or'coalign'
    OUTPUT: a mapsequece coaligned and differeneced to the first map of the sequence.

    Maps can be derotated by calculating the rotation of the center point or 
    coaligned by comparing feature. This is an alternative, and very likely a 
    better choice to the combination of rotate_coord, square_cut and map_difference.
    """
    print('Calculating mapsequence difference using', keyword)
    if keyword == 'coalign':
        coaligned = mc_coalign(mapsequence)
    elif keyword == 'derotate':
        coaligned = mc_derotate(mapsequence)
    return coaligned

def mc_difference(mapsequence, keyword='ratio'):

    map_list = []
    map_begin = mapsequence[0]
    for map in mapsequence:
        if keyword == 'ratio':
            data_diff = (map.data - map_begin.data)/map_begin.data
        elif keyword == 'plain':
            data_diff = map.data - map_begin.data
        map_diff = sunpy.map.Map(data_diff, map.meta)
        map_list.append(map_diff)
    mc_diff = sunpy.map.Map(map_list, sequence=True)

    print('Coalign and cut mapsequence ended successfully.')    
    return mc_diff


def fits_list_sum(fits_list):
    
    """
    Calculate the sum value of every fits data and note the corresponding time. 
    Returns an organized data list.
    
    First column is time string and the second is the corresponding data.
    """

    print('Calculating sum...')
    data = [[],[]]
    for fits in fits_list:
        map = sunpy.map.Map(fits)
        if map.detector == 'AIA':
            exptime = map.exposure_time.value
            datasum = np.sum(map.data) / exptime
        elif map.detector == 'HMI':
            map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi
            datasum = np.sum(map.data)
        timestr = sunpydate_to_str(map.date)
        data[0].append(timestr)
        sumfloat64 = np.float64(datasum)
        # Transform to float64, since float32 is not JSON serializable. 
        # The negative values are caused by datatype int32, add keyword dtype=int64.
        data[1].append(sumfloat64)
        # print('Sum:', datasum, 'Time:', timestr)
        
    print('Fits list sum calculation ended successfully.')
    return data


def mc_sum(mapsequence, keyword='mean'):
    
    """
    Identical function with fits_list_sum(), only that this works on mapsequence.

    Calculate the 'sum' or 'mean' value of each map in a mapsequence and note the 
    corresponding time. Returns an organized data list.
    """

    print('Calculating sum...')
    data = [[],[]]
    for map in mapsequence:
        map.data[np.where(np.isnan(map.data))] = 0 # substitute 'nan' with 0 for hmi
        # 2 mode: mean or sum. For later processing it's better to note the area of region later on.
        if keyword == 'mean':
            I = np.mean(map.data)
        elif keyword == 'sum':
            I = np.sum(map.data)
        # There is an adjustment in exposure time for aia in flare time. Cancel the effect here.
        if map.detector == 'AIA':
            I = (I / map.exposure_time).value
        elif map.detector == 'HMI':
            I = I
        timestr = sunpydate_to_str(map.date)
        data[0].append(timestr)
        sumfloat64 = np.float(I)
        # Transform to float64, since float32 is not JSON serializable.
        data[1].append(sumfloat64)
        print('Intensity:', I, 'Time:', timestr)

    print('Mapsequence sum calculation ended successfully.')
    return data


def create_output_dir(output_root=OUTPUT_PATH, dirname='output', timestamp=True):
    
    """Generate a output directory (with a timestamp). Called once a run."""

    print('Creating output path...')
    if timestamp is True:
        time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        dirname = dirname + time
    output_path = os.path.join(output_root, dirname)
    if os.path.exists(output_path):
        print('Output path already exists.')
    else:
        os.makedirs(output_path)
        print('Output path created.')

    return output_path

def sunpydate_to_str(date, format='%Y%m%dT%H%M%S'):

    """
    Trasform a sunpy time to formatted string.

    Note(2021.3.20): Now directly using datetime is prefered.
    Note(2021.4.7): Now since datetime is not JSON serializable, so time string 
    again.
    """

    time = date.to_datetime()
    time_string = datetime.datetime.strftime(time, format)
    return time_string


def write_csv(data, file_name='lightcurve_data', output_path=OUTPUT_PATH):

    """
    Save data as file_name + time + .csv in output_path.

    Note(2021.3.20): Now with timeseries and pandas dataframe, the alternative
    use of json is prefered.
    """

    time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    full_file = os.path.join(output_path, file_name + time + '.csv')
    with open(full_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)