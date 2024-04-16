from splitter import *
from threaded_f_score import *
import argparse

def fetch_parameters(filename):
    """gets the parameters for running a threaded_f_score analysis from a given file

    Args:
        filename (string): filename of file that's storing the parameters
    """
    f = open(filename)
    contents = f.readlines()
    data_file = str(contents[0].strip('\n'))
    data_rate = int(contents[1].strip('\n'))
    data_posts = contents[2].strip('\n').split(",")
    data_margin = float(contents[3].strip('\n'))
    thread_count = str(contents[4].strip('\n'))
    states = contents[5].strip('\n').split(",")
    output_file = str(contents[6].strip('\n'))
    f.close()
    return data_file, data_rate, data_posts, data_margin, thread_count, states, output_file

def run_splitter(filename, rate, posts, margin):
    """runs splitter.py to clean up data

    Args:
        filename (string): filename for data to clean
        rate (int): time between data frames in milliseconds
        posts (list): list of posts to split data on
        margin (float): how many minutes to shave off +/- from posts

    Returns:
        data: cleaned list of data blocks, ready for f score analysis
    """
    cleaned_data = Splitter(filename, rate, posts, margin)
    cleaned_data.split()
    data = cleaned_data.split_data
    return data

def run_f_score(states, blocks, thread_count):
    """runs threaded_f_score.py

    Args:
        states (list): treatment group labels
        blocks (list): list of split data
        thread_count (int): how many threads to use when processing ("max" defaults to cpu_count)

    Returns:
        ret: results from f score analysis
    """
    fsco = F_Score_Opt(states, blocks, thread_count)
    ret = fsco.get_f_score()
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "sample argument parser")
    parser.add_argument("Runner_Filename")
    args = parser.parse_args()
    filename, rate, posts, margin, threads, states, output_file = fetch_parameters(args.Runner_Filename)
    f = open(output_file, "x")
    data = run_splitter(filename, rate, posts, margin)
    results = run_f_score(states, data, threads)
    f.write(results)
    f.close()