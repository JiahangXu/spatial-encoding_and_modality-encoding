#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
import time
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--search-mark', metavar='M', type=str, default='all',
                        help='search mark of dataset', dest='searchmark')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    summary = []
    count = 0

    # participant
    for file in tqdm(os.listdir("results/")):
        if os.path.isfile(os.path.join("results", file)) is False:
            dict_mark = file # map
            # print(dict_mark)

            for file1 in os.listdir(os.path.join("results", file)):
                if os.path.isfile(os.path.join("results", file, file1)) is False:
                    model_mark = file1

                    if args.searchmark == 'all':
                        TestResult = pd.DataFrame()
                        try:
                            for csvfile in os.listdir(os.path.join("results", file, file1)):
                                if csvfile.endswith('.csv'):
                                    # print(csvfile)
                                    tempDF = pd.read_csv(os.path.join("results", file, file1, csvfile))
                                    TestResult = pd.concat([TestResult, tempDF])
                                    # print(list(tempDF))
                            summary.append([dict_mark,
                                            model_mark,
                                            str(round(np.mean(TestResult['dice_200']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_200']), 4)),
                                            str(round(np.mean(TestResult['dice_500']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_500']), 4)),
                                            str(round(np.mean(TestResult['dice_600']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_600']), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_endo'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_epi'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_epi'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_RV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_RV_endo'] if i != -1]), 4))
                                            ])
                        except:
                            pass

                    elif args.searchmark == '' and (not model_mark.endswith('LGE')) and (not model_mark.endswith('T2')):
                        TestResult = pd.DataFrame()
                        try:
                            for csvfile in os.listdir(os.path.join("results", file, file1)):
                                if csvfile.endswith('.csv'):
                                    # print(csvfile)
                                    tempDF = pd.read_csv(os.path.join("results", file, file1, csvfile))
                                    TestResult = pd.concat([TestResult, tempDF])
                                    # print(list(tempDF))
                            summary.append([dict_mark,
                                            model_mark,
                                            str(round(np.mean(TestResult['dice_200']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_200']), 4)),
                                            str(round(np.mean(TestResult['dice_500']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_500']), 4)),
                                            str(round(np.mean(TestResult['dice_600']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_600']), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_endo'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_epi'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_epi'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_RV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_RV_endo'] if i != -1]), 4))
                                            ])
                        except:
                            pass

                    elif args.searchmark != '' and model_mark.endswith(args.searchmark):

                        TestResult = pd.DataFrame()
                        try:
                            for csvfile in os.listdir(os.path.join("results", file, file1)):
                                if csvfile.endswith('.csv'):
                                    # print(csvfile)
                                    tempDF = pd.read_csv(os.path.join("results", file, file1, csvfile))
                                    TestResult = pd.concat([TestResult, tempDF])
                                    # print(list(tempDF))
                            summary.append([dict_mark,
                                            model_mark,
                                            str(round(np.mean(TestResult['dice_200']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_200']), 4)),
                                            str(round(np.mean(TestResult['dice_500']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_500']), 4)),
                                            str(round(np.mean(TestResult['dice_600']), 4)) + ' \\pm ' + str(
                                                round(np.std(TestResult['dice_600']), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_endo'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_LV_epi'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_LV_epi'] if i != -1]), 4)),

                                            str(round(np.mean([i for i in TestResult['HD_RV_endo'] if i != -1]),
                                                      4)) + ' \\pm ' + str(
                                                round(np.std([i for i in TestResult['HD_RV_endo'] if i != -1]), 4))
                                            ])
                        except:
                            pass

    cols = ["dict_mark", "model_mark",
            "Dice_LV", "Dice_myo", "Dice_RV",
            "HausDist_LVendo", "HausDist_LVepi", "HausDist_RVendo"]
    ResultsDF = pd.DataFrame(summary, columns=cols)
    outmark = '_' + args.searchmark if args.searchmark != '' else '_bSSFP'
    ResultsDF.to_csv("ResultSummary_" + time.strftime('Mar%d',time.localtime(time.time())) + outmark + '.csv')
