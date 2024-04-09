import os, csv

import random

total_samples = 200


rand_list = random.sample(range(1, 201), 40)

rand_clap_list = random.sample(rand_list, 10)
rand_nc_list = random.sample(rand_list, 10)
rand_clap_test_list = random.sample(rand_list, 10)
rand_nc_test_list = random.sample(rand_list, 10)


print(rand_list)
print(rand_clap_list)
print(rand_nc_list)
print(rand_clap_test_list)
print(rand_nc_test_list)

with open(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total\train_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total_Counting"):
        for filename in files:
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

with open(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total\valid_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total_Counting"):
        for filename in files:
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

with open(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total\test_labels.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for path, dirs, files in os.walk(r"C:\Users\David\PycharmProjects\ClapperAI\Data\Total_Counting"):
        for filename in files:
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