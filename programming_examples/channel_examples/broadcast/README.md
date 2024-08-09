# WIP: Broadcast Examples

In both of these examples, we attempt to broadcast an input `a` to 3 workers. In `single_herd`, the herd `size=[1, 3]` whereas in `multi_herd` there are 3 herds of `size=[1, 1]`.
The workers then add a unique value to each element in the input image and output the new image to a unique per-worker output.

Warning! These examples don't work, and are a work in progress.