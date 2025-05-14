import torch
import torch.nn as nn
import os
import glob
import numpy as np
from viewgcn_utils import LocalGCN, NonLocalMP, View_selector


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(),
                   os.path.join(complete_path,
                                "model-{}.pth".format(str(epoch).zfill(5))))

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))    

class ViewGCN(Model):

    def __init__(self, name, nclasses=2, num_views=24, hidden=384):
        super(ViewGCN, self).__init__(name)        

        self.nclasses = nclasses
        self.num_views = num_views
        self.hidden = hidden

        if self.num_views == 24:
            vertices = [[ 0.10351321, -0.26623718,  0.95833333],
                        [ 0.58660347,  0.17076373,  0.79166667],
                        [-0.43415312,  0.21421034,  0.875     ],
                        [-0.36647685, -0.60328982,  0.70833333],
                        [ 0.01976488, -0.9563158,   0.29166667],
                        [ 0.67752135, -0.49755606,  0.54166667],
                        [-0.15182544,  0.76571799,  0.625     ],
                        [-0.88358265, -0.09598052,  0.45833333],
                        [-0.67542972,  0.70738385,  0.20833333],
                        [ 0.61193796,  0.6963526,   0.375     ],
                        [ 0.98993913, -0.06629874,  0.125     ],
                        [-0.78018082, -0.62416487,  0.04166667],
                        [ 0.15366374,  0.98724432, -0.04166667],
                        [ 0.5497027,  -0.82595517, -0.125     ],
                        [-0.94957016,  0.23433679, -0.20833333],
                        [ 0.83957118,  0.45831298, -0.29166667],
                        [-0.29994434, -0.87715928, -0.375     ],
                        [-0.35602319,  0.81435744, -0.45833333],
                        [ 0.76855365, -0.34047396, -0.54166667],
                        [-0.73985756, -0.24896947, -0.625     ],
                        [ 0.34123727,  0.61791667, -0.70833333],
                        [ 0.1434854,  -0.59386516, -0.79166667],
                        [-0.40171157,  0.27019033, -0.875     ],
                        [ 0.28246466,  0.04255512, -0.95833333]]
        
        self.vertices = torch.tensor(vertices).cuda()

        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views, hidden=hidden)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views, hidden=hidden)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2, hidden=hidden)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2, hidden=hidden)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4, hidden=hidden)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2, hidden=hidden)
        self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4, hidden=hidden)

        self.cls = nn.Sequential(
            nn.Linear(self.hidden*3,self.hidden),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.hidden,self.hidden),
            nn.Dropout(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.hidden, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        views = self.num_views
        y = x
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)

        y = self.LocalGCN1(y,vertices)
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2,vertices,k=4)
        z = self.LocalGCN2(z,vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]

        w, F_score2, vertices3 = self.View_selector2(z2,vertices2,k=4)
        w = self.LocalGCN3(w,vertices3)
        pooled_view3 = torch.max(w, 1)[0]

        pooled_view = torch.cat((pooled_view1,pooled_view2,pooled_view3),1)
        pooled_view = self.cls(pooled_view)
        return pooled_view,F_score,F_score2
    