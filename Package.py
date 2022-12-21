#%%
from scipy.fft import fft, ifft
import cmath
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from random import random as rand
from math import atan2
import time
import math
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from tensorflow import keras


########################################################################################################################
#.....................................................Adjust SNR.......................................................#
########################################################################################################################


def adjustSNR(sig, snrdB=40, td=True):
    
    # Adds AWGN in order to provide a SNR for input signal 
    
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    noise : Vector or Tensor, optional
        Noise Tensor. The default is None.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from wav file
    sig_zero_mean = sig - sig.mean()
    psig = sig_zero_mean.var()

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        # noise = np.sqrt(pnoise)*np.random.randn(sig.shape[0], sig.shape[1] )
        noise = np.sqrt(pnoise) * np.random.normal(0, 1, sig.shape)
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        noise = real_noise + 1j * imag_noise
        noise = np.sqrt(pnoise) * abs(noise) * np.exp(1j * np.angle(noise))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise


########################################################################################################################
#..........................................Closest Value in a List.....................................................#
########################################################################################################################


def closest_value(input_list, input_value):
 
    arr = np.asarray(input_list)

    i = (np.abs(arr - input_value)).argmin()

    return arr[i]

########################################################################################################################
#.............................................Point inside a Polygon...................................................#
########################################################################################################################

def is_P_InSegment_P0P1(P, P0,P1):

# is_P_InSegement_P0P1  checks if a point lies in the borders of a polygon or not

    p0 = P0[0]- P[0], P0[1]- P[1]
    p1 = P1[0]- P[0], P1[1]- P[1]


    det = (p0[0]*p1[1] - p1[0]*p0[1])
    prod = (p0[0]*p1[0] + p0[1]*p1[1])
    
    return (det == 0 and prod < 0) or (p0[0] == 0 and p0[1] == 0) or (p1[0] == 0 and p1[1] == 0)


def isInsidePolygon(P: tuple, Vertices: list, validBorder=False) -> bool:

# isInsidePolygon   checks if a point falls inside a polygon or not
# Input Arguments
#   P             :   Source Coordinates
#   Vertices      :   Room Coordinates
#   validBorder   :   Boolean term to allow for the source coordinate to lie on the border of the polygon
# Output Arguments
#   Boolean variable that indicates if the source is within the polygon or not

    sum_ = complex(0,0)


    for i in range(1, len(Vertices) + 1):
        v0, v1 = Vertices[i-1] , Vertices[i%len(Vertices)]


        if is_P_InSegment_P0P1(P,v0,v1):
            return validBorder


        sum_ += cmath.log( (complex(*v1) - complex(*P)) / (complex(*v0) - complex(*P)) )


    return abs(sum_) > 1

 
########################################################################################################################
#...................................................Phase Extraction...................................................#
########################################################################################################################

def extract_phase(RIR, nMics, fs, lower, upper):

#   extract_phase   returns the phase function of the impulse response for all the microphones 
# Input Arguments
#   RIR         :   impulse response of nMics microphones (nMics * Nfft)
#   fs          :   sampling rate of the RIR
#   nMics       :   number of microphones in the array
#   lower       :   index of lower frequency limit
#   upper       :   index of upper frequency limit 

# Output Arguments
#   RIR_Phase   :   Phase of each frequency bin per microphone
    
    dt = 1/fs
    
    Nfft = len(RIR[0])
    
    RIR_Phase = [[0 for _ in range(np.int0(Nfft/2))] for _ in range(nMics)]
    
    for N in range(0,nMics):
        
        y = RIR[N]
        
        # Matlab FFT function.
        Y = fft(y,Nfft)
        # Only the single sided spectrum.
        Y = Y[0:int(Nfft/2)] 
        # Multiply by 2 to compensate for the single 
        # sided spectrum, and normalizing with the 
        # FFT length.

        phase_y = np.angle(2*Y*(1/Nfft)) 
        
        RIR_Phase[N] = phase_y[lower:upper]

    return RIR_Phase

########################################################################################################################
#........................Generate Index for Lower and Upper Frequency Limit............................................#
########################################################################################################################

def index_gen(Nfft, f_lower, f_upper, fs):
    
    dt = 1/fs
    
    f_vec = np.linspace(0.0, 1.0/(2.0*dt), Nfft//2)
    
    index_low = closest_value(f_vec, f_lower)

    l = np.where(f_vec == index_low)[0][0]

    index_upper = closest_value(f_vec, f_upper)

    u = np.where(f_vec == index_upper)[0][0]
    
    return l,u

########################################################################################################################
#..............................One Hot Encoding for Direction of Arrival...............................................#
########################################################################################################################

def one_hot_mapper(angle, divisions):

# one_hot_mapper    uses one hot encoding to convert the angle associated with the direction of arrival into a vector of 1s and
#                   0s with the size depending on the number of divisions (resolution of the azimuth angle).
#
# Input Arguments
#   angles      :   a vector containing azimuth angle for each run (1 * nRuns)
#   divisions   :   desired resolution of the azimuth angle (e.g. 8 divisions)
# Output Arguments
#   A           :   matrix of one hot encoded azimuth angles (divisions * nRuns)

    regions = 360/divisions

    x = np.zeros(divisions)

    for m in range(0, divisions):

        if angle >= regions*m and angle < regions*(m+1):

            x[m] = 1


    return x

########################################################################################################################
#..................................Orientation of Microphone...........................................................#
########################################################################################################################


def mic_orientation(n_mics):

#   mic_orientation a function that returns the position of a mic array in random orientations 
#
# Input Arguments
#   n_mics      :   number of microphones in the array
# Output Arguments
#   x_array     :   linearly spaced x-coordinates of mic array
#   y_array     :   linearly spaced y-coordinates of mic array
#   z_array     :   linearly spaced z-coordinates of mic array

    e = np.pi / 180 * 180 * np.random.rand()
    a = np.pi / 180 * 180 * np.random.rand()

    x = round(np.sin(a) * np.cos(e),3)
    y = round(np.cos(a) * np.cos(e),3)
    z = round(np.sin(e),3)


    x_array = np.linspace(.00, x*0.1, n_mics)
    y_array = np.linspace(.00, y*0.1, n_mics)
    z_array = np.linspace(.00, z*0.1, n_mics)

    return x_array, y_array, z_array

########################################################################################################################
#..................................Room Impulse Response Computation...................................................#
########################################################################################################################


def computeRIR(first_corner, second_corner, third_corner, fourth_corner,
              room_height, max_order, nDivisions, n_mics, rt60, snr, len_RIR, x_array, y_array, z_array,
              lower_index, upper_index):

#   computeRIR      computes the room impulse response by taking the following inputs 
#
# Input Arguments
#   first_corner     : First corner of the room
#   second_corner    : Second corner of the room
#   third_corner     : Third corner of the room
#   fourth_corner    : Fourth corner of the room
#   room_height      : Height of the room
#   max_order        : Maximum order of the ISM simulation
#   nDivisions       : Number of divisions to be made for the output 
#   n_mics           : Number of microphones in the array
#   rt60             : Reverberation Time
#   snr              : Signal to Noise Ratio
#   len_RIR          : Length of the Room Impulse Response in samples
#   x_array          : x coordinates of the mic array
#   y_array          : y coordinates of the mic array
#   z_array          : z coordinates of the mic array
#   lower_index      : index for the lower frequency limit 
#   upper_index      : index for the upper frequency limit
# Output Arguments
#   RIR_measured     : Room Impulse Response
#   RIR_Phase        : Phase Coefficients
#   angle            : Azimuth angle of the source in reference to the array
#   A                : One-hot encoded angle 
    
    roomdim = [first_corner, second_corner, third_corner, fourth_corner]

    room_coords = np.array([first_corner, second_corner, third_corner, fourth_corner]).T

    room_mag = [np.max(room_coords[0])-np.min(room_coords[0]), np.max(room_coords[1])-np.min(room_coords[1]), room_height]
    
    source_coords = (np.min(room_coords[0]) + room_mag[0] * rand(), np.min(room_coords[1]) + room_mag[0] * rand())
    while isInsidePolygon(source_coords, room_coords.T, validBorder=False)==0:
        source_coords = (np.min(room_coords[0]) + room_mag[0] * rand(), np.min(room_coords[1]) + room_mag[0] * rand())
        
    source_coords = (source_coords[0], source_coords[1], room_height/2)
    
    fs = 16000
    raytrace = False

    grid_array = np.stack([x_array, y_array, z_array], axis=0)


    # room dimensions (corners in [x,y] meters)
    room_dim = [4.0, 4.0, room_height]

    # set uniform absorption coeffs to walls
    e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
    pra.inverse_sabine(rt60, room_dim)

    # receiver centre coords
    receiver_center = np.asarray(room_dim)[:, np.newaxis] / 2

    # The locations of the microphones can then be computed as
    mic_locations = receiver_center + grid_array

    # Create the room
    room = pra.Room.from_corners(
        room_coords,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=raytrace,
        air_absorption=True)
    if raytrace:
        room.set_ray_tracing()

    # make room 3d
    room.extrude(height=room_height, materials=pra.Material(e_absorption))

    # add source to room
    room.add_source(source_coords)
    # add arrays to room
    room.add_microphone_array(mic_locations)
    
    # compute RIR
    room.compute_rir()
    
    # truncate to same length
    max_len = len(room.rir[0][0])
    for ii in range(len(room.rir)):
        if max_len < len(room.rir[ii][0]):
            max_len = len(room.rir[ii][0])
    trunc = max_len
    # split the arrays
    RIR_measured = np.zeros((n_mics, trunc))
    for ii in range(len(room.rir)):
        if ii < n_mics:
            RIR_measured[ii, :] = np.hstack((room.rir[ii][0], np.zeros((trunc - len(room.rir[ii][0], )))))
    
    RIR_measured = np.pad(RIR_measured, ((0, 0), (0, 16384 - trunc)))
    
    if snr is not None:
        RIR_measured = adjustSNR(RIR_measured, snrdB=snr)
    
    RIR_measured = RIR_measured[:n_mics, :int(len_RIR)]
        
    RIR_Phase = extract_phase(RIR_measured, n_mics, fs, lower_index, upper_index)
    
    
    # This part computes the true angle based on the coordinates of the center of the receiver (microphone) 
    # and the source. 
    
    central_pos = np.array([np.mean(mic_locations[0]), np.mean(mic_locations[1]), np.mean(mic_locations[2])])

    vert = source_coords[1] - central_pos[1]
    horz = source_coords[0] - central_pos[0]

    angle = atan2(horz, vert) * 180/np.pi
    
    if angle < 0:
    
        angle = 360 + angle
        
    angle = np.round(angle)
    
    A = one_hot_mapper(angle, nDivisions)
    
    return np.asarray(RIR_measured), np.asarray(RIR_Phase), angle, A



########################################################################################################################
#..................................Steered Response Power Functions....................................................#
########################################################################################################################

def sample_looking_angles(n_angles, z=0):

#   sample_looking_angles  converts azimuth angle into cartesian coordinates
#
# Input Arguments
#   n_angles    :   division of angles
#   z           :   z coordinate
# Output Arguments
#   [x,y,z]     :   coordinates for the looking directions

    
    length = 1.
    angle = (np.pi * np.linspace(0, 2. - 2/n_angles, n_angles) - np.pi ) * 180/np.pi

    for i in range(0,len(angle)):
        if angle[i] < 0:
            angle[i] += 360

    x = length * np.cos(angle * np.pi / 180)
    y = length * np.sin(angle * np.pi / 180)
    z = z*np.ones_like(x)
    return np.stack([x, y, z])



def fib_sphere(num_points, radius=1.):
    
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    #z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)
    z = np.zeros(num_points)
    
    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)

    x_batch = np.dot(radius, x)
    y_batch = np.dot(radius, y)
    z_batch = np.dot(radius, z)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 3)
    # plt.show()
    return np.asarray([x_batch, y_batch, z_batch])

def calc_scm(X):
    
    # X (n_freq,n_channels,n_timesteps) OR (n_freq,n_channels,n_snapshots,n_timesteps)
    if X.ndim == 4:
        return np.einsum('fnst,fmst->fnmt', X, np.conj(X), optimize=True) / X.shape[2]
    elif X.ndim == 3:
        return np.einsum('fnt,fmt->fnmt', X, np.conj(X), optimize=True)
    else:
        return np.einsum('fn,fm->fnm', X, np.conj(X), optimize=True)

def srp_map(X, A, phat=False):
    
    """
    beamforming, SRP-map via hi-mem computation, ca 1% C-time of pyroomacoustics (10% w einsum optimize=False)
    parameters:
        X (n_freq,n_channels,n_timesteps) OR (n_freq,n_channels,n_snapshots,n_timesteps)
        A
        phat (bool) use phase transform? (whitening, filtering "coloured reverberation" )
            if False, reduces to convenitonal beamforming
    output:
        P SRP power map
    """
    
    AHA = np.einsum('fnd,fmd->fnmd', np.conj(A), A)
    if phat:  # apply PHAseTransform
        X = X / np.abs(X + 1e-13)
    SCM = calc_scm(X.T)
    SRP = np.real(np.einsum('fnm,fnmd->d', SCM, AHA, optimize=True))
    # normalize by 1/2 (overlap?), n_freqs, n_mic_pairs
    SRP *= 2 / X.shape[0] / X.shape[1] / (X.shape[1] - 1)
    return SRP


def srpDOA(P, fvec, phat1, r_mic, c0=343., LookingDirections=None):
    
    if LookingDirections is None:
        LookingDirections = fib_sphere(4)  # creates unit vectors in 3D directions, Nx3
    k_abs = 2 * np.pi * fvec / c0
    A = np.exp(1j * np.einsum('i,jk,dk->ijd', k_abs, r_mic.T, LookingDirections.T))  # time delay at microphones
    powermap = srp_map(P, A, phat=phat1).astype(np.float32)
    return LookingDirections[:,np.argmax(powermap)]


def calculateSRP(IR, angles, nDivisions, n_mics, length_samp):


    angles_ref = (np.pi * np.linspace(0, 2. - 2/nDivisions, nDivisions) - np.pi ) * 180/np.pi

    for i in range(0,len(angles_ref)):

        if angles_ref[i] < 0:

            angles_ref[i] += 360

    for i in range(0, len(angles)):

        angles[i] = closest_value(angles_ref, angles[i])


    fs = 16000


    fvec = np.fft.rfftfreq(n = len(IR[0][0]), d = 1/fs)
    x_array = np.linspace(-.1, .1, n_mics)
    y_array = np.linspace(-.00, .00, n_mics)
    z_array = np.linspace(-.00, .00, n_mics)
    grid_array = np.stack([x_array, y_array, z_array], axis=0)

    looking_directions = sample_looking_angles(n_angles= 8)


    source_locations_PHAT = [0 for _ in range(length_samp)]

    for x in range(0,length_samp):

        P = np.fft.rfft(IR[x])

        a = srpDOA(P, fvec, True, grid_array, c0=343., LookingDirections=looking_directions)

        p = math.atan2(a[0], a[1]) * 180/np.pi

        if p < 0:

            p += 360

        source_locations_PHAT[x] = p


    source_locations = [0 for _ in range(length_samp)]

    for x in range(0,length_samp):

        P = np.fft.rfft(IR[x])

        a = srpDOA(P, fvec, False, grid_array, c0=343., LookingDirections=looking_directions)

        p = math.atan2(a[0], a[1]) * 180/np.pi

        if p < 0:

            p += 360

        if np.abs(p) < 1e-3:
            p = 0

        source_locations[x] = p


    srp = 0
    srp_phat = 0

    counter = 0

    for x in range (0,length_samp):

        if angles[x] <= 270 and angles[x] >= 90:
            if angles[x] == source_locations[x]:
                srp += 1
            if angles[x] == source_locations_PHAT[x]:
                srp_phat += 1
            counter+=1
    
    accuracy_srp = srp/counter * 100
    accuracy_srp_phat = srp_phat/counter * 100
    
    return accuracy_srp, accuracy_srp_phat




########################################################################################################################
#..................................Preparing Datasets for CNN Model....................................................#
########################################################################################################################


def prepare_datasets(phase_info, direction, test_size, validation_size):
      
    # create train/test split

    X_train, X_test, y_train, y_test = train_test_split(phase_info, direction, test_size=test_size)

    # create train/validation split

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (160, 4, 8192) = (nRuns, nReciever, nSamples)

    #X_train = X_train[...,np.newaxis]
    #X_validation = X_validation[...,np.newaxis]
    #X_test = X_test[...,np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


########################################################################################################################
#........................................Plotting Model Performance....................................................#
########################################################################################################################

def model_performance(history):
    
    #print(history.history.keys())

    #  "Accuracy"
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    
    
########################################################################################################################
#..................................One Hot Encoded to Angle............................................................#
########################################################################################################################
    
def one_hot_to_angle(direction_vector):
    
 
    index = np.where(direction_vector == max(direction_vector))
    index = int(index[0])

    angle = (index+1) * 360/len(direction_vector)

    if angle >= 360:
        angle = angle - 360
            
    return angle
