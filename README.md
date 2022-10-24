# GHVFDT

an implementation of Gaussian Hellinger Very Fast Decision Tree

## General overview

The main idea in this project is to find a solution for learning data streams coming from radio telescopes. The Data is imbalanced so there is one instance of the positive (Pulsar candidate i.e another planet) class every 10,000 negative instances. In other words negative class is the majority class.
The task is to propose a solution based on ML to detect these positive instances. I chose very fast decision tree model. This algorithm uses a tree learning model where the Data instances are partitioned using feature split. Heuristic measures are used to select the best spit feature.
A benefit of this approach is that the tree disables the learning process for the least promising nodes.
When the algorithm arrives at a new node of the tree, the algorithm computes the new mean and variance values of majority and minority class distributions of each feature and recomputes hellinger distance. we note the two best values and compare the difference between the two and if it is greater than an E value, the split is performed according to the best feature.

## Very Fast Decision Tree
The algorithm uses tree learning, whereby the data are partitioned using feature split point tests that aim to maximize the separation of pulsar and non-pulsar candidates. This involves first choosing the variable that acts as the best class separator, and then finding a numerical threshold ‘test point’ for that variable that maximises class separability.


<center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/45092804/197551022-1db44962-5d72-454c-9214-081b32af04e4.png" width="600" />
</p>

</center>

## The key feature of the GH-VFDT

• The use of Hellinger distance criterion for the evaluation of the split points during learning
• Only the mean and the standard deviation of the distributions of majority and minority classes of the corresponding features need to be known
• This measure prevent the classifier from becoming biased to non-pulsar class

How?
• by modelling each feature distribution as a Guassian, the distance between majority and minority distributions P and Q can be measured 
• Then getting the one that maximises the Hellinger distance between both distribution

## dataset
• class inseparability: distributions of minority class and majority class are superposed (fig. c)
• small disjuncts: distribution of minority class is composed of disjuncts small distribution (fig. d)

<center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/45092804/197558082-dd8dc3a5-545f-456e-873c-0998b0abedb2.png" width="300" />
</p>

</center>

Three databases were used. The skin dataset is collected by randomly sampling B,G,R values from face images of various age groups (young, middle, and old). Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth:
<center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/45092804/197560897-ca154caa-2357-4c9b-bbf1-b4680b7aa84e.png" width="600" />
</p>

</center>

## implementation
• Implementation of GHVFDT using the source code of Scikit-Multiflow
• New tree: ghvfdt_tree.py
• New split criterion: gaussian_hellinger_criterion.py
• New attribute observer: gh_numeric_attribute_class_observer_gaussian.py
• 3 new types of nodes: gh_active_learning_node.py, gh_learning_node_nb.py and gh_learning_node_nb_adaptive.py

• split criterion: one of the methods the criterion has is compute_Hellinger(), which calculates the Hellinger distance as in Eq.2 in the paper **Hellinger Distance Trees for Imbalanced Streams** ( https://arxiv.org/pdf/1405.2278.pdf ):

<center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/45092804/197601860-829bba6b-cfd1-4f25-8bf7-2d32de7cab77.png" width="400" />
</p>

</center>
• Attribute observer: get_best_evaluated_split_suggestions(), which returns the best split suggestion of the corresponding feature 
## training
Two functions were implemented:
• Train_tree() : for pre-training 
• Test_tree() 
• In these functions, the streams were created from the .csv files. Then, an incremental test-then-train approach was adopted as mentioned in the paper. At the end the F-score and GM were calculated
• Trying to reproduce the tables in the paper, experiments were done with the new GHVFDT and the old HDVFDT on the 3 datasets
• One experiment consists in pre-training the tree on the train set then testing it on the different test sets. 
• For each tree, this experiment is repeated 10 times with different random arrangement of the data to get average values of the F-score and GM. 

![image](https://user-images.githubusercontent.com/45092804/197604445-fd5cee6b-dcf5-446c-8809-dc178df6f1d7.png)

