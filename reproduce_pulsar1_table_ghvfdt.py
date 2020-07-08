from skmultiflow.trees import GhvfdtClassifier
from skmultiflow.data.file_stream import FileStream

import time
import copy
from pathlib import Path
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

from dataset.htru.split_htru import create_all_test_files


def train_tree(csv_path, tree):
    
    print("Training the tree")
    
    stream = FileStream(csv_path)
    
    accuracy = 0
    n_samples = 0
    correct_cnt = 0
    
    t0 = time.time()
    
    while stream.has_more_samples():
        X, y = stream.next_sample()
        y_pred = tree.predict(X)
        if y[0] == y_pred[0]:
            correct_cnt += 1
        tree = tree.partial_fit(X, y)
        n_samples += 1
    
    
    t1 = time.time()
    total = t1-t0
    
    accuracy = 100.0 * correct_cnt / n_samples
    
    print("Training data instances: ", n_samples)
    print("Tree trained on ", n_samples, " instances & has ", accuracy, "% accuracy.")
    print("Training tree completed in ", total, " (s)")


def test_tree(csv_path, tree):

    print("Testing the tree")
    
    stream = FileStream(csv_path)
    
    
    n_samples = 0
    correct_cnt = 0
    
    t2 = time.time()
    
    y_true_all = list()
    y_pred_all = list()
    while stream.has_more_samples():
        X, y = stream.next_sample()
        y_pred = tree.predict(X)
        if y[0] == y_pred[0]:
            correct_cnt += 1
        tree = tree.partial_fit(X, y)
        n_samples += 1
        
        y_true_all.append(y[0])
        y_pred_all.append(y_pred[0])
    
    
    t3 = time.time()
    total = t3-t2
    
    accuracy = 100.0 * correct_cnt / n_samples
    fscore = f1_score(y_true_all, y_pred_all, average='binary')
    gm = geometric_mean_score(y_true_all, y_pred_all, average='binary')
    
    print("Test data instances: ", n_samples)
    print("Tree tested on ", n_samples, " instances & has ", accuracy, "% accuracy.")
    print("Tree has F-score: %.3f" % fscore)
    print("Tree has GM: %.3f" % gm)
    print("Testing tree completed in ", total, " (s)")
    
    return round(fscore,3), round(gm,3)


def get_tree_results(tree, imbalancy):
    result = []
    
# create 4 copies of the tree:
    trees_copies = []
    for _ in range(4):
        trees_copies.append(copy.deepcopy(tree))

# test the tree on +1:-10 data:
    test_path = Path("dataset") / "htru" / str("test_1_"+str(imbalancy)+"_0.1.csv")
    fscore, gm = test_tree(test_path, trees_copies[0])
    result.append((fscore, gm))
    print("\n")

# test the tree on +1:-100 data:
    test_path = Path("dataset") / "htru" / str("test_1_"+str(imbalancy)+"_0.5.csv")
    fscore, gm = test_tree(test_path, trees_copies[1])
    result.append((fscore, gm))
    print("\n")

# test the tree on +1:-1000 data:
    test_path = Path("dataset") / "htru" / str("test_1_"+str(imbalancy)+"_0.75.csv")
    fscore, gm = test_tree(test_path, trees_copies[2])
    result.append((fscore, gm))
    print("\n")

# test the tree on +1:-10000 data:
    test_path = Path("dataset") / "htru" / str("test_1_"+str(imbalancy)+"_1.csv")
    fscore, gm = test_tree(test_path, trees_copies[3])
    result.append((fscore, gm))
    print("\n")
    
    return result


def run_one_test(test_idx): 
# create test_files:
    create_all_test_files(test_idx)
    
# create the trees:        
    ht = GhvfdtClassifier(binary_split = True,
                          grace_period = 200,
                          split_confidence = 0.0000001,
                          tie_threshold = 0.05,
                          split_criterion = "gaussian_hellinger")

# pretrain on 200 pos and 1000 neg:
    train_path = Path("dataset") / "htru" / "train_200_1000.csv"
    train_tree(train_path, ht)
    print("\n")

# create 4 copies of the tree:
    trees = []
    for _ in range(4):
        trees.append(copy.deepcopy(ht))
    
# do tests and get fscore and gm:
    results = []
    results.append(get_tree_results(trees[0], 10))
    results.append(get_tree_results(trees[1], 100)) 
    results.append(get_tree_results(trees[2], 1000))
    results.append(get_tree_results(trees[3], 10000))

# save results:  
    new_file_content=""
    for _, row in enumerate(results):
        new_line = str(row)[1:-1]
        new_file_content += new_line + "\n"
        
    result_file_path = Path("results") / "htru" / "GHVFDT" / str("results_"+str(test_idx)+".csv")
    with open(result_file_path, "w") as file:
        file.write(new_file_content)   
    
    return results


if __name__ == "__main__":
    
    records = [[(0,0) for _ in range(4)] for _ in range(4)]
    
    for i in range(10):
        print("iteration number:  ", i)
        results = run_one_test(i)
        for j in range(4):
            for k in range(4):
                records[j][k] = tuple(sum(x) for x in zip(records[j][k], results[j][k]))
    
    records[:] = [[tuple(map(lambda x: round(x/10, 3), j)) for j in i] for i in records]
    
    new_file_content=""
    for _, row in enumerate(records):
        new_line = str(row)[1:-1]
        new_file_content += new_line + "\n"
        
    result_file_path = Path("results") / "htru" / "GHVFDT" / str("results_average.csv")
    with open(result_file_path, "w") as file:
        file.write(new_file_content) 

  
