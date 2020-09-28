from utility import *
from argparse import ArgumentParser
import os
import re
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import h5py


def generate_foldes(train_size, valid_size, num_folds):

    data_path = os.path.join(os.path.dirname(os.getcwd()), "v3/")
    all_normal = get_image_names(os.path.join(data_path, "normal/"), label='normal/')
    all_covid = get_image_names(os.path.join(data_path, "covid/"), label='covid/')
    all_pneumonia = get_image_names(os.path.join(data_path, "pneumonia/"), label='pneumonia/')

    normal_id = list(set([re.split(r'[_]', all_normal[i])[-5] for i in range(len(all_normal))]))
    covid_id = list(set([re.split(r'[_]', all_covid[i])[-5] for i in range(len(all_covid))]))
    pneumonia_id = list(set([re.split(r'[_]', all_pneumonia[i])[-5] for i in range(len(all_pneumonia))]))

    print('NP:{}'.format(len(normal_id)))
    print('COVID:{}'.format(len(covid_id)))
    print('CAP:{}'.format(len(pneumonia_id)))

    np_valid_imgs = []
    np_train_imgs = []
    covid_valid_imgs = []
    covid_train_imgs = []
    cap_valid_imgs = []
    cap_train_imgs = []

    for i in range(num_folds):
        np_split_begin = int(np.floor(i * valid_size * len(normal_id)))
        cap_split_begin = int(np.floor(i * valid_size * len(pneumonia_id)))
        covid_split_begin = int(np.floor(i * valid_size * len(covid_id)))
        np_split_end = int(np.floor((i+1)*valid_size * len(normal_id)))
        cap_split_end = int(np.floor((i + 1) * valid_size * len(pneumonia_id)))
        covid_split_end = int(np.floor((i + 1) * valid_size * len(covid_id)))

        np_valid_list = normal_id[np_split_begin:np_split_end]
        np_train_list = normal_id[:np_split_begin]+ normal_id[np_split_end:]
        covid_valid_list = covid_id[covid_split_begin:covid_split_end]
        covid_train_list = covid_id[:covid_split_begin]+covid_id[covid_split_end:]
        cap_valid_list = pneumonia_id[cap_split_begin:cap_split_end]
        cap_train_list = pneumonia_id[:cap_split_begin]+pneumonia_id[cap_split_end:]
        np_valid_imgs.append([im for i in np_valid_list for im in all_normal if i in im])
        np_train_imgs.append([im for i in np_train_list for im in all_normal if i in im])

        covid_valid_imgs.append([im for i in covid_valid_list for im in all_covid if i in im])
        covid_train_imgs.append([im for i in covid_train_list for im in all_covid if i in im])

        cap_valid_imgs.append([im for i in cap_valid_list for im in all_pneumonia if i in im])
        cap_train_imgs.append([im for i in cap_train_list for im in all_pneumonia if i in im])

    np.save('npvalid_lists.npy', np.array(np_valid_imgs))
    np.save('nptrain_lists.npy', np.array(np_train_imgs))
    np.save('capvalid_lists.npy', np.array(cap_valid_imgs))
    np.save('captrain_lists.npy', np.array(cap_train_imgs))
    np.save('covidvalid_lists.npy', np.array(covid_valid_imgs))
    np.save('covidtrain_lists.npy', np.array(covid_train_imgs))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_size", type=int, dest="train_size",
                          default=0.8, help="training size")
    parser.add_argument("--valid_size", type=int, dest="valid_size",
                          default=0.2, help="validation and test size")
    parser.add_argument("--num_folds", type=int, dest="num_folds",
                        default=5, help="num of folds")
    args = parser.parse_args()

    generate_foldes(**vars(args))