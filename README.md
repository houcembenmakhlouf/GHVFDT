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

