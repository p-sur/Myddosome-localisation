import numpy as np
import matplotlib.pyplot as plt
#import pims
import trackpy as tp
import tifffile
import seaborn as sns
from trackpy.preprocessing import bandpass
from trackpy.find import grey_dilation
import scipy.ndimage as ndimage
import pandas as pd
import math

def track_drift(centres):
    x_initial, y_initial, z_initial = centres[0,0], centres[0,1], centres[0,2]
    deltas = []
    for row in centres:
        deltas.append([row[0] - x_initial, row[1] - y_initial, row[2] - z_initial,row[3]])
    return deltas
        
def subtract_drift(uncorrected, delta_tab):
    corrected = []
    for row in delta_tab:
        for row2 in uncorrected:
            if row[3] == row2[3]:
                xyz = row2[0:3] - row[0:3]
                t = row[3]
                intensity = row2[4]
                xyzt = np.append(xyz, t)
                xyzti = np.append(xyzt, intensity)
                xyzti = xyzti.tolist()
                corrected.append(xyzti)
    corrected = np.array(corrected)
    return corrected

def filter_tracks(traj):
    temp_traj_dict = {}
    for particle in range(0,traj['particle'].max()+1):
      temp_traj_dict[particle] = traj[traj['particle']==particle]

    trajectories_dict = {}
    for particle_track in temp_traj_dict:
        if len(temp_traj_dict[particle_track]) > 2:
          trajectories_dict[particle_track] = temp_traj_dict[particle_track]
        
    #Remove trajectories which start in frame 1 and 2
    trajectories = pd.concat(trajectories_dict.values())
    list = trajectories['particle'][trajectories['t']<=2]
    list2 = list.to_list()
    filtered_trajectories = trajectories[~trajectories['particle'].isin(list2)]
    return filtered_trajectories

def open_pointclouds(path):
    scan1 = pd.read_csv(path, names = ['x','y','z'])

    z_min = scan1['z'].min()
    z_max = scan1['z'].max()

    x_coords = list(scan1[scan1['z']==z_min]['x'])
    y_coords = list(scan1[scan1['z']==z_min]['y'])
    z_coords = list(scan1[scan1['z']==z_min]['z'])

    #x_coords = []
    #y_coords = []
    #z_coords = []

    for a in range(z_min+1, z_max):
        m = np.zeros((500,500))
        filter_list = scan1[scan1['z']==a]
        nd = filter_list.to_numpy()

        for i in nd:
            m[i[0],i[1]] = 1

        m1 = ndimage.binary_erosion(m)
        m2 = m-m1


        for j in range(len(m2)):
            for k in range(len(m2[j])):
                if m2[j][k] == 1:
                    y_coords.append(k)
                    x_coords.append(j)
                    z_coords.append(a)


    x_coords = x_coords + list(scan1[scan1['z']==z_max]['x'])
    y_coords = y_coords + list(scan1[scan1['z']==z_max]['y'])
    z_coords = z_coords + list(scan1[scan1['z']==z_max]['z'])

    z_coords_scaled = [i*(200/106.7) for i in z_coords]
    outline_coordinates = {'x': x_coords, 'y':y_coords, 'z': z_coords_scaled}
    outlines_cloud = pd.DataFrame(outline_coordinates)
    pointcloud_array = outlines_cloud.to_numpy()
    return pointcloud_array

def nearest_distance(point, point_cloud_array):
    distances = [math.sqrt((point[0]-p[0])**2 + (point[1]-p[1])**2 + (point[2]-p[2])**2) for p in point_cloud_array]
    distances.sort()
    return distances[0]


#def filter_bad_tracks(unfiltered_distance_array):

def add_empty_column_2darray(twodarray):
    length = len(twodarray)
    zeros = np.zeros((length,1))
    empty_distances = np.concatenate((twodarray, zeros), axis = 1)
    return empty_distances

def get_initial_membrane_distance(dataframe):
    #convert dataframe into dictionary, track ID as key
    unique_track = dataframe['track number'].unique()
    dict = {}
    for i in unique_track:
        dict[i] = dataframe[dataframe['track number']==i] 
        
    lifetime = []
    track_ID = []
    first_frame = []
    final_frame = []
    initial_distance_from_membrane = []
    total_displacement = []
    velocity = []
    
    for i in dict:
        first_frame_temp = dict[i].iloc[0]['t']
        final_frame_temp = dict[i].iloc[len(dict[i])-1]['t']
        lifetime_temp = final_frame_temp - first_frame_temp + 1
        initial_distance = dict[i].iloc[0]['distance']
        final_distance = dict[i].iloc[len(dict[i])-1]['distance']
        displacement = final_distance - initial_distance
        #positive displacement = movement away from membrane (into cytoplasm), negative displacement = movement towards the membrane
        velocity_temp = displacement/lifetime_temp
        #velocity in pixels/frame (frame = 10s, pixel = 0.1067 um)
        
        lifetime.append(lifetime_temp)
        track_ID.append(int(i))
        first_frame.append(first_frame_temp)
        final_frame.append(final_frame_temp)
        initial_distance_from_membrane.append(initial_distance)
        total_displacement.append(displacement)
        velocity.append(velocity_temp)
        
    updated_dict = {'trackID': track_ID, 'first frame': first_frame, 'final frame': final_frame, 'lifetime/frames': lifetime, 'initial distance from membrane/px': initial_distance_from_membrane, 'total displacement/px': total_displacement, 'velocity/px/frame': velocity}
    return pd.DataFrame(updated_dict)

def drop_bad_tracks(dataframe):
    v2_data = dataframe.drop(dataframe[dataframe['final frame'] >= 198].index)
    v2_data = v2_data.drop(v2_data[v2_data['final frame'] == 99].index)
    v2_data = v2_data.drop(v2_data[v2_data['first frame'] == 100].index)
    v2_data = v2_data.drop(v2_data[v2_data['first frame'] == 101].index)
    return v2_data

def separate_membrane_initialised_tracks(df, i):
    filtered_membrane_bound = df.drop(df[df['initial distance from membrane/px'] > i].index)
    filtered_cytoplasmic = df.drop(df[df['initial distance from membrane/px'] <= i].index)
    return filtered_membrane_bound, filtered_cytoplasmic


def pixel_number(array):
    size = 1
    for dim in np.shape(array): 
        size *= dim
    return size


def snr(coords, radius,  mod_image):

    xmax, xmin = coords[0] + radius[0], coords[0] - radius[0]
    ymax, ymin = coords[1] + radius[1], coords[1] - radius[1]
    zmax, zmin = coords[2] + radius[2], coords[2] - radius[2]

    x_r = np.arange(int(xmin),int(xmax)+1,1)
    y_r = np.arange(int(ymin),int(ymax)+1,1)
    z_r = np.arange(int(zmin),int(zmax)+1,1)

    X,Y,Z = np.meshgrid(x_r,y_r,z_r)


    spot_intensity = mod_image[Z,Y,X]
    #spot_bkg_intensity = image_bp[Z_big, Y_big, X_big]
    
    I_spot = np.sum(spot_intensity)/pixel_number(spot_intensity)
    #I_bkg = np.sum(spot_bkg_intensity)/(pixel_number(spot_bkg_intensity)-pixel_number(spot_intensity))
    I_full_background = np.mean(mod_image)

    #contrast = (I_spot - I_bkg)/(I_spot + I_bkg)
    #SNR = (I_spot - I_bkg)/np.std(spot_intensity) ## another definition
    SNR = I_spot/I_full_background
    return SNR, I_spot


def snr_row(row, image):
    (x,y,z) = row[2], row[1], row[0]
    (x1,y1,z1) = row[6], row[5], row[4]
    row['SNR'], row['mean I'] = snr((x,y,z), (x1,y1,z1), image)
    return row


#new snr calc here
def simple_SNR(row, image):
    spot_volume_cuboid = (2*row[4])*(2*row[5])*(2*row[6])
    average_spot_intensity = row[3]/spot_volume_cuboid
    SNR = average_spot_intensity/np.mean(image)
    return SNR

def extract_features(image, d, m, p):
    time_length = image.shape[0] #length of video (normally 198)
    
    separation = tuple([x + 1 for x in (11,5,5)]) #
    size = [int(2*s/np.sqrt(3)) for s in separation] #from trackpy documentation
    
    bandpass_dict = {}
    for j in range(time_length):
        temp_image = bandpass(image[j],1,d)
        bandpass_dict[j] =  ndimage.grey_dilation(temp_image, size, mode = 'constant')
        print(j)
    
    feature_dict = {}
    for i in range(time_length):
        feature = tp.locate(image[i], diameter = d, minmass = m, percentile = p)
        feature = feature.dropna(subset = ['z','y','x'])
        feature_dict[i] = feature.apply(snr_row, image = bandpass_dict[i], axis=1) 
        print(i)
    
    return (feature_dict, bandpass_dict)

                       
def calc_frame_averages(bpass_dict):
    mean_intensity = [np.mean(bpass_dict[i]) for i in bpass_dict]
    std_intensity = [np.std(bpass_dict[i]) for i in bpass_dict]
    index = [i for i in bpass_dict]
    image_data = pd.DataFrame({'frame': index,
                              'mean': mean_intensity,
                              'std': std_intensity})
    return image_data
    
def check_threshold(threshold_factor, features_dict, image_data):
    plt.figure(figsize = (10,75))
    for i in features_dict:
        mean = image_data.loc[image_data['frame']==i,'mean'].values[0]
        std = image_data.loc[image_data['frame']==i,'std'].values[0]
        intensities = features_dict[i]['mean I']
        plt.subplot(66,3,i+1)
        plt.hist(intensities, bins = 50)
        plt.axvline(mean + threshold_factor*std, color = 'r')
    plt.show()
        
def apply_threshold(threshold_value, features_dict, image_data):
    thresholded_dict = {}
    for i in features_dict:
        if len(features_dict[i]) >= 1:
            mean = image_data.loc[image_data['frame']==i,'mean'].values[0]
            std = image_data.loc[image_data['frame']==i,'std'].values[0]
            threshold = mean + threshold_value*std
            thresholded_dict[i] = features_dict[i][features_dict[i]['mean I'] >= threshold]
    return thresholded_dict
        
def combine_dict(thresholded_dict):
    for i in thresholded_dict:
        frame = [(i+1) for l in range(len(thresholded_dict[i]))]
        thresholded_dict[i].insert(3, 't', frame)
    all_locs = [thresholded_dict[i] for i in thresholded_dict]
    loc_table = pd.concat(all_locs)
    return loc_table  

def new_nearest_distances(traj_df, pointcloud_dict):
    traj_array = traj_df.to_numpy()
    add_empty_column_2darray(traj_array)
    for row in traj_array:
        z = row[0]
        y = row[1]
        x = row[2]
        t = row[3]
        pointcloudt = pointcloud_dict[t]
        distances = [math.sqrt((x-p[0])**2 + (y-p[1])**2 + (z-p[2])**2) for p in pointcloudt]
        distances.sort()
        row[5] = distances[0]
        
def df_to_dict_track(dataframe):
    #convert dataframe into dictionary, track ID as key
    unique_track = dataframe['particle'].unique()
    dict = {}
    for i in unique_track:
        dict[i] = dataframe[dataframe['particle']==i] 
    return dict
    
def membrane_bound_only(track_dict, membrane_dist):
    membraneonly_tracks = {}
    for i in track_dict:
        if track_dict[i].iloc[0,5] <= membrane_dist:
            membraneonly_tracks[i] = track_dict[i]
    return membraneonly_tracks


def combinedict_to_df(dict):
    all = [dict[i] for i in dict]
    return pd.concat(all)

def fill_in_blank_jumps(dict):
    new_dict = {}
    for i in dict:
        dict1 = dict[i]
        t_i = int(dict1.iloc[0,3])
        t_f = int(dict1.iloc[-1,3])
        t_lifetime = int(t_f - t_i + 1)
        track_full = np.zeros((t_lifetime, 3))
        for t in range(t_i, t_f + 1):
            if t in dict1['t'].unique():
                track_full[t-t_i, 0] = t
                track_full[t-t_i, 1] = dict1.loc[dict1['t'] == t, 'distance'].item()
            else:
                t_temp = t
                while t_temp not in dict1['t'].unique():
                    t_temp = t_temp - 1
                track_full[t-t_i, 0] = t
                track_full[t-t_i, 1] =  dict1.loc[dict1['t'] == t_temp, 'distance'].item()
            track_full[t-t_i,2] = i
        new_dict[i] = pd.DataFrame(track_full, columns = ['t', 'distance', 'particle'])
    return new_dict

def jd_list(df, membrane_position):
    empty_arr = np.zeros((198,3))
    for t in range(198):
        membrane_bound = df.loc[((df['t'] == t+1) & (df['distance'] < membrane_position)), 'particle'].to_numpy()
        cytoplasm = df.loc[((df['t'] == t+1) & (df['distance'] > membrane_position)), 'particle'].to_numpy()
        empty_arr[t,0] = t+1
        empty_arr[t,1] = len(membrane_bound)
        empty_arr[t,2] = len(cytoplasm)
    complete_df = pd.DataFrame(empty_arr, columns = ['t/frames', 'membrane bound', 'cytoplasm'])
    return complete_df    



parameters = {
    #3: 5,
    #4: 4.8, 
    #5: 4.8,
    #6: 4.8,
    7: 4.8,
    9: 4.8,
    11: 4.8,
    12: 4.5,
    13: 4.2
}

for r in parameters:

    image = tifffile.imread("C:/Users/klene/Desktop/prasanna/0714/only gaussian individual cells/cell" + str(r) + ".tif")
    feature_dict, bandpass_dict = extract_features(image, (11,5,5), 2000, 75)
    frame_parameters = calc_frame_averages(bandpass_dict)
    threshold_dict = apply_threshold(parameters[r], feature_dict, frame_parameters)
    loclist = combine_dict(threshold_dict)
    traj = tp.link(loclist, search_range = 11, pos_columns = ['z','y','x'], t_column = 't', memory = 5)

    filtered_traj = filter_tracks(traj)
    filtered_traj = filtered_traj.drop(columns = ['mass', 'size_z', 'size_y', 'size_x', 'ecc', 'signal', 'raw_mass', 'ep_z', 'ep_y', 'ep_x', 'SNR', 'mean I'] )
    filtered_traj['z'] = filtered_traj['z']*(200/106.7)
    filtered_traj = filtered_traj.iloc[:,[2,1,0,3,4]]


    pointcloud_dict = {}
    for i in range(1, 199):
        pointcloud_dict[i] = open_pointclouds("C:/Users/klene/Desktop/prasanna/0714/results1/cell" + str(r) + "-" + str(i) + ".csv")
        print(i)

    traj_array = filtered_traj.to_numpy()
    empty_distances = add_empty_column_2darray(traj_array)
    for i in range(0,len(empty_distances)):
        t = int(empty_distances[i,3])
        coordinates = empty_distances[i,0:3]
        extract_from_dict = pointcloud_dict[t]
        minimum_distance = nearest_distance(coordinates, extract_from_dict)
        empty_distances[i,5] = minimum_distance
        print(i)

    distances = pd.DataFrame(empty_distances, columns = ['x','y','z','t','particle','distance'])
    track_dict = df_to_dict_track(distances)
    if len(track_dict) >= 1:
        membrane_only = membrane_bound_only(track_dict, membrane_dist = 14)
        if len(membrane_only) >= 1:
            membrane_only2 = fill_in_blank_jumps(membrane_only)
            membrane_only_df = combinedict_to_df(membrane_only2)
            membrane_only_df = membrane_only_df.sort_values(by = 't')
            plot = jd_list(membrane_only_df, 14)
            plot.to_csv("C:/Users/klene/Desktop/prasanna/0714/jd output/cell" + str(r) + ".csv")
