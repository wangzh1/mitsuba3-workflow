import drjit as dr

class MSE:
    def __call__(self, x, y):
        return dr.mean(dr.sqr(x - y))