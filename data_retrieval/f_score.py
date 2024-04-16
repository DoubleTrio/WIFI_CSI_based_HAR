import bisect
import numpy as np

class F_Score:
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
    
    def get_f_score(self):
        """for each column in the data set, this splits the column at the pre-determined index
            and will then find the t-score for the two arrays from the split
        """
        if self.split_idx is None: #if you don't manually get the index, it will just do it for you
            self.get_temporal_idx()
        arr_a = []
        arr_b = []
        data = np.swapaxes(self.data, 0, 1) #swap row and column indexes for easier navigation
        for i in range(len(data)):
            print("Channel", i)
            arr_a = data[i][0:self.split_idx]
            arr_b = data[i][self.split_idx:]
            mean_a = np.mean(arr_a)
            mean_b = np.mean(arr_b)
            var_a = np.var(arr_a)
            var_b = np.var(arr_b)
            f_score = ((mean_a-mean_b)**2)/(var_a+var_b)
            self.f_scores.append(f_score)
        return self.f_scores