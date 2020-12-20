# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:38:19 2020

@author: s150127
"""
#import sys
import numpy as np
from pims import ND2_Reader # reader of ND2 files
from pims import Frame # for converting ND2 generator to one numpy array
import time
import multiprocessing as mp

import sys
sys.path.append('../')
#sys.path.insert(1, 'C:\Users\s150127\OneDrive - TU Eindhoven\Werk\MBx Python analysis\Code')

import _code.fitters as fitting
import _code.roi_finding as roi_finding
import cProfile

pr = cProfile.Profile()

#%% Init
ROI_SIZE = 9
WAVELENGTH = 637 #nm
filenames = ("C:/Users/s150127/Downloads/_MBx dataset/1nMimager_newGNRs_100mW.nd2",)

n_processes = int(mp.cpu_count()/4)


#%% Main

def main(frames, metadata, roi_locations, min_intensity, q):
    
    # ND2 = ND2_Reader(name)
    # frames = ND2
    # metadata = ND2.metadata
    # frames[0:2]
    
    fitter = fitting.phasor_only_ROI_loop(metadata, ROI_SIZE, WAVELENGTH, min_intensity, "Phasor")
    
    print("fitter")
    
    # for roi_index, roi in enumerate(roi_locations):
    #         y = int(roi[0])
    #         x = int(roi[1])
    #         print("xy")
    #         frame_stack = frames[:,y-ROI_size_1D:y+ROI_size_1D+1, x-ROI_size_1D:x+ROI_size_1D+1]
    #         print("stack")
    
    test = fitter.main(frames, metadata, roi_locations)
    
    print("Done fiting")
    
    #roi_locations = roi_locations*2
    
    
    
    q.put(test)


#%% Main
if __name__ == '__main__':
    
    #pr.enable()

    for name in filenames:
        with ND2_Reader(name) as ND2:

            ## parse ND2 info
            frames = ND2
            metadata = ND2.metadata
            #frames = frames[0:2]
            
            frame_stack_total = Frame(frames)
            
            
            roi_finder = roi_finding.roi_finder(ROI_SIZE, frames[0])#, intensity_min = 800)
            fitter = fitting.scipy_last_fit_guess(metadata, ROI_SIZE,
                                                  WAVELENGTH, roi_finder.intensity_min, 
                                                  "ScipyLastFitGuess", 5)
            roi_locations = roi_finder.main(frames[0], fitter)
            
            #roi_locations = roi_locations[0:4, :]
            
            start = time.time()
            processes=[]
            q = mp.Queue()
            for i in range(0, n_processes):
                roi_locations_split = np.array_split(roi_locations, n_processes)
                processes.append(mp.Process(target=main, args=(frame_stack_total, metadata, roi_locations_split[i], 
                                                               roi_finder.intensity_min, q)))

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            result = [q.get() for p in processes]
            
            print("received")
            
            first = True
            for res in result:
                if first == True:
                    results = res
                    first = False
                else:
                    results = np.vstack((results, res))
            print('Time taken: ' + str(round(time.time() - start, 3)) + ' s. Fits done: ' + str(results.shape[0]))
            print(results)
            #pr.disable()
            #pr.print_stats(sort='time')
