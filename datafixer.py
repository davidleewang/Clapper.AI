import os, csv
import random


# manually enter the total number of samples and the fraction of training samples
num_total_samples = 1000
idx_total_samples = num_total_samples + 1

train_fraction = 0.8
valid_test_fraction = 1 - train_fraction

num_valid_test_samples = round(num_total_samples * valid_test_fraction)

# list of samples to be put into valid/test splits
rand_list = random.sample(range(1, idx_total_samples), num_valid_test_samples)

# these are the lists for validation labels

num_valid_samples = round(num_valid_test_samples / 2)

num_clap_valid_samples = round(num_valid_samples / 2)
num_nc_valid_samples = num_valid_samples - num_clap_valid_samples

rand_clap_list = random.sample(rand_list, num_clap_valid_samples)
rand_nc_list = random.sample(rand_list, num_nc_valid_samples)

print('This is the number of validation claps:', len(rand_clap_list))
print('This is the number of validation no-claps:', len(rand_nc_list))

# these are the lists for test labels
num_test_samples = num_valid_test_samples - num_valid_samples

num_clap_test_samples = round(num_test_samples / 2)
num_nc_test_samples = num_test_samples - num_clap_test_samples

rand_clap_test_list = random.sample(rand_list, num_clap_test_samples)
rand_nc_test_list = random.sample(rand_list, num_nc_test_samples)

print('This is the number of test claps:', len(rand_clap_test_list))
print('This is the number of test no-claps:', len(rand_nc_test_list))

with open(r"Data\Total\train_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"Data\Total"):
        for filename in files:
            if filename.endswith('.wav'):
                justname = filename.replace('.wav', '')

                exclude = False
                if 'clap' in filename:
                    label = 1
                    num = justname[4:]
                    for compare_num in rand_clap_list:
                        if num == str(compare_num):
                            exclude = True

                    for compare_num in rand_clap_test_list:
                        if num == str(compare_num):
                            exclude = True

                else:
                    label = 0
                    num = justname[2:]
                    for compare_num in rand_nc_list:
                        if num == str(compare_num):
                            exclude = True

                    for compare_num in rand_nc_test_list:
                        if num == str(compare_num):
                            exclude = True

                if exclude != True:
                    writer.writerow([filename, label])

with open(r"Data\Total\valid_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"Data\Total"):
        for filename in files:
            if filename.endswith('.wav'):
                justname = filename.replace('.wav', '')

                include = False
                if 'clap' in filename:
                    label = 1
                    num = justname[4:]
                    for compare_num in rand_clap_list:
                        if num == str(compare_num):
                            include = True

                else:
                    label = 0
                    num = justname[2:]
                    for compare_num in rand_nc_list:
                        if num == str(compare_num):
                            include = True

                if include == True:
                    writer.writerow([filename, label])

with open(r"Data\Total\test_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"Data\Total"):
        for filename in files:
            if filename.endswith('.wav'):
                justname = filename.replace('.wav', '')

                include = False
                if 'clap' in filename:
                    label = 1
                    num = justname[4:]
                    for compare_num in rand_clap_test_list:
                        if num == str(compare_num):
                            include = True

                else:
                    label = 0
                    num = justname[2:]
                    for compare_num in rand_nc_test_list:
                        if num == str(compare_num):
                            include = True

                if include == True:
                    writer.writerow([filename, label])