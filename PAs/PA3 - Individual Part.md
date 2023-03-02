# Question 1:

## Part 1
##### Filter 1
| X | X  | X  | X | X | 
|---|---|---|---|---|
| X | -1 | 2 | 2 | X | 
| X | -2  | 2 | 4 | X | 
| X | -2 | 2  | 5 | X | 
| X | X  | X  | X | X | 
##### Filter 2
| X | X | X | X | X |
|---|---|---|---|---|
| X | -1 |-4 | -2 | X |
| X | -1 |-2 | -3 | X |
| X |1 | 5 | 2 | X |
| X | X | X | X | X |
##### Output Feature Map
| X | X  | X  | X | X |
|---|----|----|---|---|
| X | -2  | -2 | 0 | X |
| X | -1 | 0 | 1 | X |
| X | -1 | 7  | 7 | X |
| X | X  | X  | X | X |

## Part 2
The first one detects the vertical gradient, and the second one detects the horizontal gradient (or edge) on the input feature

## Part 3
##### Input Subpatch 1
| 1 | X | -1 |
|---|---|---|
| 1 | X | -1 |
| X | X | -1 |
##### Input Subpatch 2
| 1 | 1 | 1 |
|---|---|---|
| X | X | X |
|-1 | X |-1 |

## Part 4
##### Spatial Max Pooling
| 0 | 1 |
|---|---|
| 7 | 7 |

## Part 5
##### Filter 1
2
##### Filter 2
-3
##### Output Feature Map
-1

## Part 6
##### (a) Filter feature shape
I use \[in channels, out channels, height, width\] in terms of PyTorch convention
- **conv1**: \[3, 5, 5, 5\]
- **conv2**: \[5, 10, 7, 7\]
- **conv3**: \[10, 1, 13, 13\]

Since both the input images and kernels are square, without loss of generality, we let $I,f$ stands for the number of features per dimension per input, or filter, respectively. Thus, the filter shape can be described as the flooring of $[(I-f_{h})/S]^2\times O$ where $f_h=(f-1)/2$, $S$ stands for stride, and $O$ stands number of output channels. So, by taking the numbers into the formula we get:
##### (b) **conv1** output feature shape
- $252\times252\times5$
##### (c) **conv2** output feature shape
- $246\times246\times10$
##### (d) **maxpool** output feature shape
- $122\times122\times10$
##### (e) **conv3** output feature map shape
- $11\times11\times1$

##### (f) Number of input and output units of fc1 and fc2 layer
- **fc1**: $(11\times11)\times512=121\times512=61952$ units 
- **fc2**: $512\times10=5120$ units

