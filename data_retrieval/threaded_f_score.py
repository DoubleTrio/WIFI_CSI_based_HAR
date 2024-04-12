from multiprocessing import Pool
import numpy as np

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
    
    def mapFunc(self, row):
        """find the sample size, mean, and variance of a given set of data

        Args:
            row (list): the set of data to find the aforementioned statistics for

        Returns:
            nA, xA, vA: returns the sample size, mean, and variance as calculated by Numpy
        """
        return (np.size(row), np.mean(row), np.var(row))
    
    def reduceFunc(self, rows):
        """combines the mean and variance of two different smaller samples by taking weighted combinations

        Args:
            row1 (list): block of data to combine with another block
            row2 (list): same as row 1 but for a second sample set

        Returns:
            xAB: combined mean of the two samples
            vAB: combined variance of the two samples
        """
        (row1, row2) = rows
        nA, xA, vA = self.mapFunc(row1) #row 1 is made up of the sample size, mean, and variance of A
        nB, xB, vB = self.mapFunc(row2) #sample sizes are prefixed with 'n', means with 'x', and variances with 'v'
        nAB = nA + nB
        xAB = ((xA * nA) + (xB * nB)) / nAB
        vAB = (((nA * vA) + (nB * vB)) / nAB) * ((xB - xA) / nAB)**2
        return (xAB, vAB)
    
    def get_f_score(self):
        """finds the f scores for all data block pairs
        """
        
        for i in range(len(self.pairs)): #loop through all pairs
            for j in range(len(self.blocks[0][0])): #loop through all channels
                priorA = self.statistics.get(self.labels[i][0][j])
                if priorA:
                    xA, vA = priorA
                priorB = self.statistics.get(self.labels[i][1][j])
                if priorB:
                    xB, vB = priorB

                if not priorA and not priorB: #redundancy prevention
                    total_lines = len(self.pairs[i][0]) + len(self.pairs[i][1])
                    parallel_processes = min(total_lines, self.thread_count)
                    pool = Pool(processes=parallel_processes) #set up parallel threads
                    
                    if not priorA:
                        chunkA = np.array_split(self.pairs[i][0][j], 2)
                        inp = (chunkA[0], chunkA[1])
                        chunks = np.array_split(chunkA, self.thread_count)
                        #print(chunks)
                        ret = pool.map(self.reduceFunc, [[0, 1],[1, 2]])
                        xA, vA = ret
                        stat = (xA, vA)
                        label = self.labels[i][0] + str(j)
                        self.statistics.update({label:stat})
                    
                    if not priorB:
                        chunkB = np.array_split(self.pairs[i][1][j], self.thread_count)
                        chunks = np.array_split(chunkB, 2)
                        inp = (chunks[0], chunks[1])
                        xB, vB = pool.map(self.reduceFunc, inp)
                        
                        stat = (xB, vB)
                        label = self.labels[i][1] + str(j)
                        self.statistics.update({label:stat})
                    
                f_score = ((xA-xB)**2)/(vA+vB) #final f score for the channel
                self.pair_scores.append((self.labels[i] + str(j), f_score))
        return self.pair_scores