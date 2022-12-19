# tree-nat
The idea of this project is that the time complexity of a balanced binary tree is O(〖log〗_2 N), 
so that the target sentence can be generated in a tree structure, which can effectively improve the decoding speed. 
In the tree structure generation process, the generation of target tokens at each level can use the previously generated tokens as a reference, 
thus effectively improving the correlation between each target token.


Specifically, the target sentence itself is a sequence, first insert a <BOS> token in the middle of the sentence, convert it into a binary tree structure, 
and ensure that the two child nodes of each node in the tree are valid tokens or <EOS> tokens; then mask A certain level of the binary tree, 
the level of the masked tokens can only see the tokens in the upper level that are considered to have been generated, 
and the tokens in the lower level are considered to have not been generated, so they are invisible. 
Finally, the binary tree is converted into sequences by levels for training. In the prediction stage, 
each level of the tree structure is auto-regressively generated from the root node of the <BOS> token until the predicted number of levels is reached 
or a certain level of <EOS> token is generated, and then restored to the sequence target sentence.
