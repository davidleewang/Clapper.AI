import torch

from model import CustomMobileNetV2, save_model
from utils import load_data, accuracy

import csv

# to start tensorboard: (run in console)
#   tensorboard --logdir (your desired output folder in the hw directory)

# to train model:
#   python -m train 15 10 0.05

def train(args):

    model = CustomMobileNetV2()

    if torch.cuda.is_available():
        d = "cuda"
    else:
        d = "cpu"

    device = torch.device(d)

    model = model.to(device)

    epochs = args.epoch

    data_path = "Data/Total/"
    valid_path = "valid_labels.csv"
    # test_path = "test_labels.csv"
    train_path ="train_labels.csv"

    train_data = load_data(data_path, train_path, num_workers=0, batch_size=args.batch_size, augmentation_flag=True)

    valid_data = load_data(data_path, valid_path, num_workers=0, batch_size=5, augmentation_flag=False)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max')

    global_step = 0
    report_strings = []
    training_summary_list = []
    for epoch in range(1, epochs+1):
        model.train()
        epoch_accuracy_history = []

        for image, label in train_data:
            # print(image)
            # print(label)

            processed_image = image

            # print(processed_image.shape)
            if device is not None:
                processed_image, label = processed_image.to(device), label.to(device)


            predicted_value = model(processed_image)
            loss = criterion(predicted_value, label)

            batch_accuracy_value = accuracy(predicted_value, label)
            epoch_accuracy_history.append(batch_accuracy_value.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            # scheduler.step()

        average_epoch_accuracy = sum(epoch_accuracy_history) / len(epoch_accuracy_history)
        # print(average_epoch_accuracy)


        # dev logging

        model.eval()

        valid_accuracy_history = []

        with torch.no_grad():
            for image, label in valid_data:

                processed_image = image

                if device is not None:
                    processed_image, label = processed_image.to(device), label.to(device)

                predicted_valid_value = model(processed_image)
                batch_accuracy_value = accuracy(predicted_valid_value, label)
                valid_accuracy_history.append(batch_accuracy_value.detach().cpu().numpy())

            average_valid_accuracy = sum(valid_accuracy_history) / len(valid_accuracy_history)

        report_string = f'epoch {epoch} \t acc = {round(average_epoch_accuracy, 4)} \t val acc = {round(average_valid_accuracy, 4)}'
        report_strings.append(report_string)

        print(report_string)

        training_summary_list.append((epoch, average_epoch_accuracy, average_valid_accuracy))

        save_model(model.pretrained_model.classifier[1], epoch)

        with open(r"C:\Users\David\PycharmProjects\ClapperAI\training_outputs\training_summary.csv", 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['Epoch', 'Avg Train Acc', 'Avg Valid Acc'])
            for summary_item in training_summary_list:
                epoch = summary_item[0]
                avg_ep_acc = round(summary_item[1], 4)
                avg_val_acc = round(summary_item[2], 4)
                writer.writerow([epoch, avg_ep_acc, avg_val_acc])

    print('\n')
    for s in report_strings:
        print(s)
    print('\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('--log_dir')
    # Put custom arguments here

    parser.add_argument('epoch', type=int, default=15)

    parser.add_argument('batch_size', type=int, default=10)

    parser.add_argument('learning_rate', type=float, default=0.05)

    args = parser.parse_args()
    train(args)
