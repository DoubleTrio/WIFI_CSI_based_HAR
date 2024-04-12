from splitter import *
from threaded_f_score import *
import argparse






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "sample argument parser")
    parser.add_argument("user")
    args = parser.parse_args()
    if args.user == "me":
        print("test complete")
    else:
        print("huh? wuh huh?")