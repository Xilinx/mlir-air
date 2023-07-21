import numpy as np

ROW1 = 9
COL = 256

inp1 = np.zeros((ROW1*COL,))
for i in range(ROW1*COL):
    inp1[i] = i

out = np.zeros((ROW1*COL,))
out_a = np.zeros((ROW1*COL,))

ROW = 5
for r1 in range(3):
    inp = inp1[r1*COL:(r1+ROW)*COL]
    for c in range(2, COL-2):
        for r in range(2, ROW-2):
            index = c+r*COL
            lap_ij =  4*inp[index] - inp[index - COL] - inp[index + COL] - inp[index - 1] - inp[index + 1]
            lap_imj = 4*inp[index - COL] - inp[index - 2*COL] - inp[index] - inp[index - 1 - COL] - inp[index + 1 - COL]
            lap_ipj = 4*inp[index + COL] - inp[index] - inp[index + 2*COL] - inp[index - 1 + COL] - inp[index + 1 + COL]
            lap_ijm = 4*inp[index - 1] - inp[index - 1 - COL] - inp[index - 1 + COL] - inp[index - 2] - inp[index]
            lap_ijp = 4*inp[index + 1] - inp[index + 1 - COL] - inp[index + 1 + COL] - inp[index] - inp[index + 2]

            flx_ij = (lap_ipj - lap_ij) 
            if flx_ij * (inp[index + COL] - inp[index]) > 0:
                flx_ij = 0
            
            flx_imj = (lap_ij - lap_imj) 
            if flx_imj * (inp[index] - inp[index - COL]) > 0:
                flx_imj = 0

            fly_ij = (lap_ijp - lap_ij) 
            if fly_ij * (inp[index + 1] - inp[index]) > 0:
                fly_ij = 0

            fly_ijm = (lap_ij - lap_ijm) 
            if fly_ijm * (inp[index] - inp[index - 1]) > 0:
                fly_ijm = 0
            
            out[index+(r1*COL)] = inp[index] - 7*(flx_ij - flx_imj + fly_ij - fly_ijm)

print(out[2*COL:3*COL])
print(out[3*COL:4*COL])
print(out[4*COL:5*COL])