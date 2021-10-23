
class AsymetricLoss:
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=-0.05, eps=1e-8):
        super(AsymetricLoss, self).__init__()