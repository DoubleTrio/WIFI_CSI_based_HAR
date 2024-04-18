import bisect
import numpy as np

class Splitter:
    def __init__(self, filename, rate, posts, margin):
        self.filename = filename #the input CSV file
        self.rate = rate #the time between each data reading (in milliseconds)
        self.posts = posts #the timestamps where conditions were altered (in minutes since experiment's start)
        self.margin = margin #the amount of time surrounding the Posts to cut from the data (in minutes)
        self.timestamps = [] #time indexes for the data lines
        self.split_data = [] #the split data with each cleaned block being one of the indexes
        self.data = []
        
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
        
    def split(self):
        """
        Takes the data from the input file and breaks it up into cleaned/clipped blocks for ease of use in other code
        """
        index_margin = int(( self.margin * 60 * 1000 ) / self.rate) #converts margin to indexes for easier data clean-up
        prior = index_margin
        for post in self.posts:
            post = float(post)
            print("post",post)
            post_ms = post * 60 * 1000
            print("post_ms",str(post_ms),"+ self.timestamps[0]",str(self.timestamps[0]),"=",str(post_ms+self.timestamps[0]))
            post_ms += self.timestamps[0]
            idx = bisect.bisect_left(self.timestamps, post_ms) #bin search for the post's location
            print("Post", post, " Found @ Index", idx)
            block = self.data[prior:(idx-index_margin)] #creation of the clipped data block
            prior = idx+index_margin #updating the prior for the next block
            if prior > len(self.data): #failsafe in case one of the post's is way too close to the end of data collection
                prior = len(self.data) - 1              #this data should not be used, but the program should still run
            block = np.swapaxes(block, 0, 1)
            self.split_data.append(block)
        block = self.data[prior:(len(self.data)-index_margin)] #makes final block of data (which technically does not have an associated post)
        block = np.swapaxes(block, 0, 1)
        self.split_data.append(block)