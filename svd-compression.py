import sys
import numpy 

from PIL import Image

class SVD_factored: 
    A: numpy.matrix = None
    U: numpy.matrix = None
    S: numpy.matrix = None
    Vt: numpy.matrix = None

    # Rank
    r: int = None

    def __init__(self, matrix = None, U = None, S = None, Vt = None): 
        # If given a matrix, perform Single Value Decomposition on that matrix
        if (matrix):
            self.A = numpy.asmatrix(matrix)
            (self.U, self.S, self.Vt) = numpy.linalg.svd(self.A)
        # If given Single Value Decomposition, generate matrix formed by the decomposition
        else:
            self.U = U
            self.S = S
            self.Vt = Vt

            self.A = self.U * numpy.diag(self.S) * self.Vt
        
    def SVD_Compact(self):
        # Calc rank if not already calculated
        if (not self.r):
            self.r = numpy.linalg.matrix_rank(self.A)
        
        # Truncate zero values
        U = self.U.transpose()[0:self.r].transpose()
        S = self.S[0:self.r]
        Vt = self.Vt[0:self.r]

        # Create and return new svd object
        svdc = SVD_factored(U = U, S = S, Vt = Vt)
        svdc.r = self.r
        return svdc

    def SVD_Truncate(self, n):
        # Calc rank if not already calculated
        if (not self.r):
            self.r = numpy.linalg.matrix_rank(self.A)

        # If n is greater than the rank of the matrix, use rank instead of N because the same result is achieved. 
        if (n > self.r):
            return self.SVD_Truncate(self.r)
        
        # Truncate matrices to compress
        _U = self.U.transpose()[0:n].transpose()
        _S = self.S[0:n]
        _Vt = self.Vt[0:n]

        # Create and return new svd object
        svdt = SVD_factored(U = _U, S = _S, Vt = _Vt)
        return svdt


def svdToRBG(red: SVD_factored, green: SVD_factored, blue: SVD_factored):
    if (red.A.shape != green.A.shape or green.A.shape != blue.A.shape):
        print("Dimensions dont match up!!")
        return None
    
    # Combine rbg arrays into one 3d array. 
    shape = red.A.shape
    return [[[red.A[i, j], green.A[i, j], blue.A[i, j]] for j in range(shape[1])] for i in range(shape[0])]

if __name__ == "__main__": 
    if (len(sys.argv) >= 2):
        img = Image.open(sys.argv[1])
        imgarr = numpy.asarray(img)

        print("Parsing image and generating Compact SVD...")
        
        # Perform SVD on rbg values
        imgsvd_r = SVD_factored([[j[0] for j in i] for i in imgarr])
        imgsvd_g = SVD_factored([[j[1] for j in i] for i in imgarr])
        imgsvd_b = SVD_factored([[j[2] for j in i] for i in imgarr])

        # Compact SVD
        imgsvdc_r = imgsvd_r.SVD_Compact()
        imgsvdc_g = imgsvd_g.SVD_Compact()
        imgsvdc_b = imgsvd_b.SVD_Compact()

        print("Finished Compact SVD. Saving... ")

        # Save Compact SVD to file
        svdc_A_rgb = numpy.uint8( svdToRBG(imgsvdc_r, imgsvdc_g, imgsvdc_b))
        svdc = Image.fromarray(svdc_A_rgb)
        svdc.save(f"{sys.argv[1]}_svdc.png")

        print(f"Saved to \"{sys.argv[1]}_svdc.png\"")

    if (len(sys.argv) >= 3):

        print(f"Generating Truncated(n={sys.argv[2]}) SVD...")

        # Truncate SVD to specified amount
        imgsvdt_r = imgsvd_r.SVD_Truncate(int(sys.argv[2]))
        imgsvdt_g = imgsvd_g.SVD_Truncate(int(sys.argv[2]))
        imgsvdt_b = imgsvd_b.SVD_Truncate(int(sys.argv[2]))

        print("Finished Truncated SVD. Saving... ")

        # Save Truncated SVD to file
        svdt_A_rgb = numpy.uint8( svdToRBG(imgsvdt_r, imgsvdt_g, imgsvdt_b))
        svdt = Image.fromarray(svdt_A_rgb)
        svdt.save(f"{sys.argv[1]}_svdt{sys.argv[2]}.png")

        print(f"Saved to \"{sys.argv[1]}_svdt{sys.argv[2]}.png\".")
