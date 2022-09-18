# fuse-conv-and-bn

refer to repvgg

fuse conv and bn to conv 

function: (in utils.py)
1. fuse conv and bn to conv 
2. transform conv_kX to conv_kY    ;;ps: conv_kX means  kernel is X
3. transform identity block to conv_kY 
