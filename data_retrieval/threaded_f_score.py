import bisect
import numpy as np

class F_Score_Opt:
    def __init__(self, filename, timestamp):
        self.filename = filename #filename for where the data set is stored
        self.timestamp = timestamp #timestamp to temporally split data on
        self.timestamps = [] #timestamps from data set
        self.data = [] #data from data set with each row having an associated timestamp
        self.split_idx = None
        self.f_scores = []
        
        f = open(filename)
        contents = f.readlines()
        for line in contents:
            split_arr = line.split(",")
            self.timestamps.append(split_arr[1]) #extract just timestamps to its own array
            middle = np.asfarray(split_arr[2:],dtype=float)
            self.data.append(middle) #don't need channel count or timestamp in the raw data array
        f.close()
        self.timestamps = np.asfarray(self.timestamps,dtype=float)
        self.data = np.array(self.data)
        
    def get_temporal_idx(self):
        """uses the given timestamp in initialization to find the index of where the data should be split
        Returns
        -------
        index of the value closest to the initialized timestamp (rounded up)
        """
        idx = bisect.bisect_left(self.timestamps, self.timestamp) #binary search for the index so its slightly faster (hopefully)
        if self.timestamps[idx] < self.timestamp:
            idx += 1
        self.split_idx = idx
        return idx
    
    def mapFunc(self, row):
        """find the sample size, mean, and variance of a given set of data

        Args:
            row (list): the set of data to find the aforementioned statistics for

        Returns:
            nA, xA, vA: returns the sample size, mean, and variance as calculated by Numpy
        """
        return (np.size(row), np.mean(row), np.var(row))
    
    def reduceFunc(self, row1, row2):
        """combines the mean and variance of two different smaller samples by taking weighted combinations

        Args:
            row1 (list): 3 element list containing the first sets: sample size, mean, and variance
            row2 (list): same as row 1 but for a second sample set

        Returns:
            xAB: combined mean of the two samples
            vAB: combined variance of the two samples
        """
        nA, xA, vA = self.mapFunc(row1) #row 1 is made up of the sample size, mean, and variance of A
        nB, xB, vB = self.mapFunc(row2) #sample sizes are prefixed with 'n', means with 'x', and variances with 'v'
        nAB = nA + nB
        xAB = ((xA * nA) + (xB * nB)) / nAB
        vAB = (((nA * vA) + (nB * vB)) / nAB) * ((xB - xA) / nAB)**2
        return xAB, vAB
    
    def get_f_score(self):
        """for each column in the data set, this splits the column at the pre-determined index
            and will then find the t-score for the two arrays from the split
        """
        if self.split_idx is None: #if you don't manually get the index, it will just do it for you
            self.get_temporal_idx()
        arr_a = []
        arr_b = []
        data = np.swapaxes(self.data, 0, 1) #swap row and column indexes for easier navigation
        for column in data:
            arr_a = column[0:self.split_idx] #sample set 1
            arr_b = column[self.split_idx:] #sample set 2
            
            # Using "reduceFunc" and threading, determine
            # the mean and variance of the two sets of 
            # data and calculate their fischer score
            
            f_score = ((xA-xB)**2)/(vA+vB) #final f score for the channel
            self.f_scores.append(f_score)
        return self.f_scores