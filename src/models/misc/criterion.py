import drjit as dr

class MSE:
    def __call__(self, x, y):
        return dr.mean(dr.sqr(x - y))
    
class L1:
    def __call__(self, x, y):
        return dr.mean(dr.abs(x - y))

class L1Smooth:
    def __call__(self, x, y, beta=1.0):
        diff = dr.abs(x - y)
        return dr.mean(dr.select(diff < beta, 
                                 0.5 * dr.sqr(diff) / beta, 
                                 diff - 0.5 * beta))
        