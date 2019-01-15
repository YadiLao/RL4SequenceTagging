# MM-Tag
This is an implementation of MM-Tag which is a MCTS enhanced MDP model. More detail can be seen at https://arxiv.org/abs/1804.10911
The dataset we used is CoNLL2003 (Name Entity Recognition)

cnn-lstm: MM-Tag with char feature.
new-value: MM-Tag with new implementation of value network. 
policy-gradient: the policy-gradient model under the same MDP setting for sequence labeling.

Evaluation metric:
Entity Level F1 score. note that in MM-Tag, label level F1 is our reward.

Useage:

1. run torch_process_data.py to generate data.
2. run mcts_simple_go.py / torch_mcts.py / policy_gradient.py in corresponding folder.

Environment:
python3
treelib
tensorflow
pytorch
