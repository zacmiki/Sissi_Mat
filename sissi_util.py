#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:36 2022

SISSI-MAT useful functions

getListOfFiles(dirName) --- Returns a list of all files in the given directory "dirName"
allOpusFiles(dirName) --- returns a list with all the OPUS files in that dir and all subdirs
loadandgraph(fileName, graph, params) -- load the filename and returns the data, graph and params are not zero they are displayed

savitzky_golay(y, window_size, order, deriv=0, rate=1) --- returns the sav-gol smoothed array

DACPress(wl) - Returns the pressure in GPa - given the ruby line wavelength in nm
DACTemp(wl) -- Returns the temperature in K given the position of ruby line wl in nm
DACLinePos(T) - TO IMPLEMENT

wn2En(waveNumber) -- Returns the Energy in meV from the value in cm-1
En2wn(waveNumber) -- Returns the WveNumber from the energy in meV
wn2THz(wavenumber) - wavenumber to terahertz converter
THz2wn(terahertz) - terahertz to wavenumber converter

parValues(fileName) -- returns the meaningful Parameters of the OPUS file passed to the procedure

@author: miczac
"""
#------------------------------------------
def getListOfFiles(dirName):
    import os

    # Create an empty list to store the names of the files
    fileList = []

    # Use the os.walk function to iterate over the files and directories in the specified directory
    for root, dirs, files in os.walk(dirName):
        # Add the names of the files in the current directory to the file list
        for file in files:
            fileList.append(os.path.join(root, file))

    return fileList

#------------------------------------------

def allOpusFiles(dirName):
    '''Returns a list with all the OPUS files in that dir and all subdirs'''
    import opusFC
    allFiles = getListOfFiles(dirName)
    opusFiles = list()

    for entry in allFiles:
        if opusFC.isOpusFile(entry):
            opusFiles.append(entry)

    return opusFiles

#------------------------------------------

def test_opus(all_files):
    '''Returns a list with all the OPUS files in that dir and all subdirs'''
    import opusFC

    # list comprehension to create a list with just the opus_files
    opusFiles = [entry for entry in all_files if opusFC.isOpusFile(entry)]

    return opusFiles

#------------------------------------------

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    '''returns the sav-gol smoothed array'''
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError:
        print("window_size and order have to be of type int")

    except window_size % 2 != 1 or window_size < 1:
        print("window_size size must be a positive odd number")

    except window_size < order + 2:
        print("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself

    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#------------------------------------------

def DACPress(wlength):
    '''Returns the pressure in GPa - given the ruby line wavelength in nm'''
    A = float(1904)
    B = float(7.715)
    R1 = float(694.19)

    A1 = float(1870)
    B1 = float(5.63)

    pressure1 = A/B * ((wlength/R1)**B -1)
    pressure2 = A1*((wlength-R1)/R1)*(1 + B1*((wlength-R1)/R1))

    print(f"\nFor a Ruby wavelength of {wlength} nm the pressure is %.3f  {pressure1} GPa")
    print(f"By Using the Ruby2020 formula the pressure is %.3f {pressure2} GPa\n")
    return pressure1

#------------------------------------------

def DACTemp(wlength):
    '''Returns the temperature in K given the position of ruby line wl in nm'''
    Tambient = float(298.1) #Our Ambient temperature while acquiring the ref ruby line
    R1 = float(694.19)      #Our Reference ruby Line position at Ambient Temperature

    # Tambient = float(input("\n \nEnter the value of the ambient temp in K: "))   ---- old
    print(f"\nRef wavelength used: {R1} nm\nAmbient Temperature used: {Tambient} K")

    #T1 = float(273.15 + Tambient)
    T2 = (wlength - R1)/0.00726 + Tambient

    print("For a wavelength of", wlength, "the temperature is %.3f" % T2, "K")
    return

#------------------------------------------
def DACLinePos(temp):
    Tambient = float(298.1) #Ref Ambient temperature while acquiring the ref ruby line
    R1 = float(694.19)      #Ref ruby Line position at Ambient Temperature
    print("\nRef wavelength and temperature for our ruby are ", R1, "nm at %.3f" % Tambient, "K")

    R2 = R1 + (temp - Tambient) * 0.00726

    print(f"At {temp}K the line will be at %.3f" % R2, "nm\n")
    return

#------------------------------------------

def wn2en(wavenumber):
    h = 4.135667516E-15
    c = 299792458
    wavenum = float(wavenumber)
    energy = h * c * wavenum * 100 * 1000
    print(f'{wavenumber} cm-1 => = ', energy , 'meV')
    return

#------------------------------------------

def parvalues(fileName):
    '''returns the meaningful Parameters of the OPUS file passed to the procedure'''
    import opusFC
    import matplotlib.pyplot as plt
    import os

    dbs = opusFC.listContents(fileName)

    for item in range(len(dbs)):
        if (dbs[item][0]) != 'SIFG':
            data = opusFC.getOpusData(fileName, dbs[item])
            print("\n\033[1m Acquisition Parameters \n\033[0m")

            print(f"Spec Type =\t{data.parameters['PLF']}")
            print(f"Aperture =\t{data.parameters['APT']} \nBSplitter = \t{data.parameters['BMS']}")
            print(f"Source = \t{data.parameters['SRC']} \nDetector = \t{data.parameters['DTC']}")
            print(f"Frequency = \t{data.parameters['VEL']} kHz\nChannel = \t{data.parameters['CHN']}")
            print(f"Resol = \t{data.parameters['RES']} cm-1")
            print(f"Data from:\t{data.parameters['LXV']} to {data.parameters['FXV']} cm-1")
            print(f"Pressure =\t{data.parameters['PRS']} hPa")
            print(f"Acq Date =\t{data.parameters['DAT']}")
            print(f"Acq Time =\t{data.parameters['TIM']}")
    return

#------------------------------------------

def thz2wn(freq2convert):
    '''terahertz to wavenumber converter'''
    convFactor = 33.356
    waveNumber = freq2convert * convFactor
    print(freq2convert, 'THz ==>',  waveNumber, 'cm-1')
    return

#------------------------------------------

def wn2thz(wavenumber):
    convFactor = 33.356
    freq = wavenumber / convFactor
    print(wavenumber, 'cm-1 ==>',  freq, 'THz')
    return

#------------------------------------------

def loadandgraph(fileName, graphornot = None, paramornot = None):
    '''Load the filename and returns the data, graph and params are not zero they are displayed'''
    import opusFC
    import matplotlib.pyplot as plt
    import os

    dbs = opusFC.listContents(fileName)
    print(f"\nFile {os.path.basename(fileName)} Loaded\n")
    print(dbs, "\n")

    for item in range(len(dbs)):
        if (dbs[item][0]) != 'SIFG':
            data = opusFC.getOpusData(fileName, dbs[item])
            labella = os.path.basename(fileName) + "_" + dbs[item][0]
            suffix = dbs[item][0]

            # If you want to print to graph of not the opus file
            if graphornot == None:
                fig, ftir1 = plt.subplots()  # Create a figure containing a single axis.
                ftir1.minorticks_on()
                ftir1.set(xlabel='Wavenumbers (cm-1)', ylabel='Intensity', title= labella)
                ftir1.plot(data.x, data.y, label = suffix, linewidth= 0.5)  # Plot IR Transformed spectrum.
                ftir1.legend()

                ftir1.grid(which = 'both', axis = 'x', lw = .2)
                ftir1.grid(which = 'major', axis = 'y', linewidth = .2)
                plt.show()

            # If you want to print the files parameters
            if paramornot == None:
                print(f"\n\033[1m{labella} Acquisition Parameters \n\033[0m")
                print(f"Spec Type =\t{data.parameters['PLF']}")
                print(f"Aperture =\t{data.parameters['APT']} \nBSplitter = \t{data.parameters['BMS']}")
                print(f"Source = \t{data.parameters['SRC']} \nDetector = \t{data.parameters['DTC']}")
                print(f"Frequency = \t{data.parameters['VEL']} kHz\nChannel = \t{data.parameters['CHN']}")
                print(f"Resol = \t{data.parameters['RES']} cm-1")
                print(f"Data from:\t{data.parameters['LXV']} to {data.parameters['FXV']} cm-1")
                print(f"Pressure =\t{data.parameters['PRS']} hPa")
    return data

    #------------------------------------------

def loadSSC(fileName):
    import opusFC
    import os

    dbs = opusFC.listContents(fileName)
    print(f"File {os.path.basename(fileName)} Loaded {dbs}")

    for item in range(len(dbs)):
        if (dbs[item][0]) == 'SSC':
            data = opusFC.getOpusData(fileName, dbs[item])

    return data
