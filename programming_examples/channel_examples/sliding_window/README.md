
# WIP: Sliding Window of Data Example Using Channels

This example shows how to access a sliding window of data using channels.

It is a work in progress because ideally you wouldn't need to read in the data all at once.

## Overview

Goals: Input in one large block
```
[width, height] = [4, 8]
example:
0 0 0 0
1 1 1 1
2 2 2 2
3 3 3 3
4 4 4 4
5 5 5 5
6 6 6 6
7 7 7 7
```

Output: 3 rows added to each other
```
[width, height] = [4, 8 - (window_size - 1)]
tile_size = [1, 4]
```
Example:
```
3 3 3 3 (rows: 0 1 2)
6 6 6 6 (rows: 1 2 3)
9 9 9 9 (rows: 2 3 4)
12 12 12 12 (rows: 3 4 5)
15 15 15 15 (rows: 4 5 6)
18 18 18 18 (rows: 5 6 7)
```
