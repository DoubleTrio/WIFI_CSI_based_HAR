import multiprocessing as mp
import numpy as np
import tqdm

# References:
# https://thedeadbeef.wordpress.com/2010/06/16/calculating-variance-and-mean-with-mapreduce-python/
# https://qxf2.com/blog/run-your-api-tests-in-parallel-mapreducepython/

# Planned/Notes:
# 64 or 128 threads
# opt/research/dense_pose

class F_Score_Opt:
    def __init__(self, states, blocks, thread_count):
        self.states = states #labels for the blocks of data
        self.blocks = blocks #array of data blocks to be compared
        self.thread_count = thread_count #how many threads to use for parallel processing
        self.labels = [(a,b) for idx, a in enumerate(self.states) for b in self.states[idx + 1:]] #sets up labels for data pairs    
        self.pairs = [(a,b) for idx, a in enumerate(self.blocks) for b in self.blocks[idx + 1:]] #sets up our data pairs for comparison
        self.pair_scores = [] #list of fischer scores for each pair of data, tuple'd with combined labels ie ((labels), score)
        self.statistics = {} #dictionary for recording statistics so we don't rerun mean and variance on a pair multiple times
    
    def set_mean_stat(self, chunk):
        """find the sample size and sum of a given set of data

        Args:
            chunk (list): the set of data to find the aforementioned statistics for

        Returns:
            nA, xA, vA: returns the sample size and sum
        """
        return (np.sum(chunk), len(chunk))
    
    def set_var_stat(self, chunk, mean):
        """finds the sum of squares for a given chunk of data

        Args:
            chunk (list): set of data to find the sum of squares of
            mean (float): mean of the chunk
        """
        return np.sum((chunk-mean)**2)

    def set_mean_var_stat(self, data, pool):
        """finds the mean and variance of a given data set using parallel processing

        Args:
            data (list): set of data to find the mean and variance of
            
        Rets:
            x (float): mean of data set
            v (float): variance of data set
        """ 
        #determine suitable chunk size for parallel processing
        chunk_size = int(np.ceil(len(data)/self.thread_count))
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        #find sums and sample size of chunks of data
        ret = pool.map(self.set_mean_stat, chunks)
        
        xt, nt = sum(result[0] for result in ret), sum(result[1] for result in ret)
        x = xt / nt #summed sums divided by summed sample sizes = overall mean
        
        #find sum of squares for chunks using parallel processing
        part_sos = pool.starmap(self.set_var_stat, [(chunk, x) for chunk in chunks])
        
        sost = sum(part_sos)
        v = sost / nt
        
        return x, v
    
    def get_f_score(self):
        """finds the f scores for all data block pairs
        """
        if self.thread_count == "max":
            num = mp.cpu_count()
            pool = mp.Pool(processes=num)
            self.thread_count = num
        else:
            self.thread_count = int(self.thread_count)
            pool = mp.Pool(processes=self.thread_count)
        
        for i in range(len(self.pairs)): #loop through all pairs
            print("Pair", i)
            for j in range(len(self.blocks[0][0])): #loop through all channels
                print("Channel", j)
                priorA = self.statistics.get((self.labels[i][0],j))
                if priorA:
                    xA, vA = priorA
                priorB = self.statistics.get((self.labels[i][1],j))
                if priorB:
                    xB, vB = priorB

                if not priorA and not priorB: #redundancy prevention
                    if not priorA:
                        xA, vA = self.set_mean_var_stat(self.pairs[i][0][j], pool)
                        stat = (xA, vA)
                        label = (self.labels[i][0],j)
                        self.statistics.update({label:stat})
                    
                    if not priorB:
                        xB, vB = self.set_mean_var_stat(self.pairs[i][1][j], pool)
                        stat = (xB, vB)
                        label = (self.labels[i][1],j)
                        self.statistics.update({label:stat})
                    
                f_score = ((xA-xB)**2)/(vA+vB) #final f score for the channel
                lab = (self.labels[i][0], self.labels[i][1], j)
                self.pair_scores.append((lab, f_score))
        return self.pair_scores