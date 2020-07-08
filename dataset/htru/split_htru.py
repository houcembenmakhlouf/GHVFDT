import random
import pathlib


def extract_pos_neg_data(random_idx):
    pos_data = []
    neg_data = []
    current_file_path = pathlib.Path(__file__).parent.absolute()
    dataset_path = current_file_path / "SPINN.csv"
    with open(dataset_path, "r") as file:
        line = file.readline()
        line = file.readline()
        while line:
            stripped_line = line.strip()
            anno = stripped_line.split(",")
            if int(anno[8])==1:
                pos_data.append(anno)
            else:
                anno[8]=0
                neg_data.append(anno)
            line = file.readline()

    # shuffle data:
    # random.seed(random_idx)
    # random.shuffle(pos_data)
    # random.shuffle(neg_data)

    return pos_data, neg_data


def create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx):

    random.seed(random_idx)
    data = random.sample(pos_data, n_pos) + random.sample(neg_data, n_neg)
    
    # random.shuffle(data)

    new_file_content = "A,B,C,D,E,F,G,H,class\n"

    with open(file_path, "w") as file:
        for _, anno in enumerate(data):
            new_line = str(anno[0])+','+str(anno[1])+','+str(anno[2])+','+\
                       str(anno[3])+','+str(anno[4])+','+str(anno[5])+','+\
                       str(anno[6])+','+str(anno[7])+','+str(anno[8])
            new_file_content += new_line + "\n"

        file.write(new_file_content)


def create_test_sets(pos_data, neg_data, labelling_pct, random_idx):
# apply labelling pourcentage:
    pos_data = pos_data[:int(len(pos_data)*labelling_pct)]
    neg_data = neg_data[:int(len(neg_data)*labelling_pct)]

# get current file path:
    current_file_path = pathlib.Path(__file__).parent.absolute()
# create test file +1:-10:
    if len(pos_data) * 10 > len(neg_data):
        n_neg = len(neg_data)
        n_pos = int (n_neg / 10)
    else: 
        n_pos = len(pos_data)
        n_neg = int (n_pos * 10)        
    file_path = current_file_path / str("test_1_10_"+str(labelling_pct)+".csv")
    create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx)


# create test file +1:-100:
    if len(pos_data) * 100 > len(neg_data):
        n_neg = len(neg_data)
        n_pos = int (n_neg / 100)
    else: 
        n_pos = len(pos_data)
        n_neg = int (n_pos * 100) 
    file_path = current_file_path / str("test_1_100_"+str(labelling_pct)+".csv")
    create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx)

# create test file +1:-1000:
    if len(pos_data) * 1000 > len(neg_data):
        n_neg = len(neg_data)
        n_pos = int (n_neg / 1000)
    else: 
        n_pos = len(pos_data)
        n_neg = int (n_pos * 1000) 
    file_path = current_file_path / str("test_1_1000_"+str(labelling_pct)+".csv")
    create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx)

# create test file +1:-10000:
    if len(pos_data) * 10000 > len(neg_data):
        n_neg = len(neg_data)
        n_pos = int (n_neg / 10000)
    else: 
        n_pos = len(pos_data)
        n_neg = int (n_pos * 10000)
    if n_pos == 0: n_pos = 1
    file_path = current_file_path / str("test_1_10000_"+str(labelling_pct)+".csv")
    create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx)


def create_all_test_files(random_idx):
    # convert data to csv:
    # convert_data_to_csv()

# extract positiv and negativ data:
    pos_data, neg_data = extract_pos_neg_data(random_idx)

# create train file +200:1000:
    n_pos = 200
    n_neg = 1000
    current_file_path = pathlib.Path(__file__).parent.absolute()
    file_path = current_file_path / "train_200_1000.csv"
    create_sub_set(file_path, pos_data, neg_data, n_pos, n_neg, random_idx)

    pos_data = pos_data[n_pos:]
    neg_data = neg_data[n_neg:]

# create sub set according labelling pourcentage:
    create_test_sets(pos_data, neg_data, 0.1, random_idx)
    create_test_sets(pos_data, neg_data, 0.5, random_idx)
    create_test_sets(pos_data, neg_data, 0.75, random_idx)
    create_test_sets(pos_data, neg_data, 1, random_idx)

if __name__ == "__main__":
    create_all_test_files(0)




