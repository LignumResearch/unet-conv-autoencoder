import argparse
from lib import UNet, adaboost, dummy, rbf_svm, linear_svm, rf
from time import localtime, strftime
import os
import subprocess


def main():
    
    ################### ARGUMENT PARSING #############################
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '-p', type=str, required=False, default="softwoods",
    #                     help="image dataset path, defaults to \'softwoods\'")
    parser.add_argument('--split', '-s', type=float, required=False, default=.05,
                        help="ratio of training images to test images, defaults to 0.05")
    parser.add_argument('--batch_size', '-b', type=int, required=False, default=32,
                        help="set batch size, defaults to 32")
    parser.add_argument('--epochs', '-e', type=int, required=False, default=10000,
                        help="maximum number of epochs to train for, defaults to 10000")
    parser.add_argument('--r_patience', '-rp', type=int, required=False, default=35,
                        help="ReduceLROnPlateau patience, defaults to 35")
    parser.add_argument('--e_patience', '-ep', type=int, required=False, default=40,
                        help="EarlyStopping patience, defaults to 40")
    # parser.add_argument('--output', type=str, required=False, default="")
    args = parser.parse_args()

    # Create output folder
    ofolder = "UNet_results_{}".format(strftime("%Y%m%d_%H%M%S", localtime()))
    os.mkdir(ofolder)
    os.chdir(ofolder)

    paths = ['softwoods', 'hardwoods']
    for path in paths:
        os.mkdir(path)    
        os.chdir(path)
        latent, _, train = UNet.train(**vars(args), path=path)
    
        print("=====passing to dummy classifier=====")
        dummy.fit(latent, train)
        print("=====done=====")
        
        print("=====passing to adaboost classifier=====")
        adaboost.fit(latent, train)
        print("=====done=====")
        
        print("=====passing to rbf svm classifier=====")
        rbf_svm.fit(latent, train)
        print("=====done=====")
        
        print("=====passing to linear svm classifier=====")
        linear_svm.fit(latent, train)
        print("=====done=====")
        
        print("=====passing to random forest classifier=====")
        rf.fit(latent, train)
        print("=====done=====")
        
        os.chdir('..')
    
    subprocess.run(['Rscript', '../metrics.py'])


if __name__ == "__main__":
    main()