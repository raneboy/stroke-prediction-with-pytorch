import torch
from torch.utils.data import Dataset, DataLoader
from csv import reader


###########################################################################################
# j - column
# j0 id - skip
# j1 Gender : Male - 1 ; Female - 0
# j2 age ; j3 hypertension ; j4  heart_disease
# j5 Ever_married : Yes - 1 ; No - 0
# j6 Work_type : Private - 1 ; Self-employed - 2 ; Govt_job - 3
# j7 Residence_type : Urban - 1 ; Rural - 2
# j8 avg_glucose_level
# j9 BMI : N/A - 0
# j10 Smoking_status : Formerly smoke - 1 ; Smokes - 2 ; Never smoked - 3 ; Unknown - 0
# j11 stroke
###########################################################################################


# Process the data
# Change the string to specific integer
# Remove unimportant column data
def process_data_to_array(path):
    processed_input_data = []

    with open(path, 'r') as file_read:
        csv_reader = reader(file_read)

        # Loop Each row of data from csv
        for i, row in enumerate(csv_reader):

            # Skip first row
            if i == 0:
                continue
            row_data = []

            # Loop each column
            for j, column in enumerate(row):

                # Skip first id column
                if j == 0:
                    continue

                # Gender
                elif j == 1:
                    if column == 'Male':
                        row_data.append(1)
                    else:
                        row_data.append(0)

                # Ever Married
                elif j == 5:
                    if column == 'Yes':
                        row_data.append(1)
                    else:
                        row_data.append(0)

                # Work Type
                elif j == 6:
                    if column == 'Private':
                        row_data.append(1)
                    elif column == 'Self-employed':
                        row_data.append(2)
                    elif column == 'Govt_job':
                        row_data.append(3)
                    else:
                        row_data.append(0)

                # Residence Type
                elif j == 7:
                    if column == 'Urban':
                        row_data.append(1)
                    elif column == 'Rural':
                        row_data.append(2)
                    else:
                        row_data.append(0)

                # BMI
                elif j == 9:
                    if column == 'N/A':
                        row_data.append(0)
                    else:
                        row_data.append(float(column))

                # Smoking Status
                elif j == 10:
                    if column == 'formerly smoked':
                        row_data.append(1)
                    elif column == 'smokes':
                        row_data.append(2)
                    elif column == 'never smoked':
                        row_data.append(3)
                    else:
                        row_data.append(0)

                else:
                    row_data.append(float(column))

            processed_input_data.append(row_data)

            # if i == 10:
            #    break

    return processed_input_data


# Split the dataset into train and test set based on ratio
def split_dataset(dataset, train_set_ratio=0.6):
    length_of_dataset = len(dataset)
    train_set = []

    # Loop until the i = ratio * length_of_dataset
    i = 0
    while i < int(length_of_dataset * train_set_ratio):
        random_index = torch.randint(len(dataset), size=(1,)).item()
        train_set.append(dataset[random_index])
        dataset.pop(random_index)
        i = i + 1

    test_set = dataset
    return train_set, test_set


# Create a custom dataset which can be iterable by dataloader
class CustomStrokeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        length = len(self.dataset[0])
        label = self.dataset[idx][-1]               # Last value in single row array
        features = self.dataset[idx][0:length-1]    # All value except last element
        return torch.tensor(features), torch.tensor(label)


# Demonstration
# The process of preparing data from csv to dataloader(ready to train)
def demonstrate():
    data = process_data_to_array('./data/healthcare-dataset-stroke-data.csv')
    train, test = split_dataset(data)
    datasets = CustomStrokeDataset(train)
    train_dataloader = DataLoader(datasets, batch_size=64, shuffle=True)
    for batch, (x, y) in enumerate(train_dataloader):
        print("Batch ", batch)
        print(x)
        print(y)
        if batch == 1:
            break


# demonstrate()
