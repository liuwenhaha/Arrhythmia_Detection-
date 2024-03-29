"""
===========
SMOTE + ENN
===========

An illustration of the SMOTE + ENN method.

"""

# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN



url="dataset.csv"

import numpy as np
y=np.array([7,14,1,1,1,10,3,1,10,6,1,1,10,1,1,1,1,1,1,1,1,16,14,10,2,2,6,1,1,1,4,1,1,10,1,6,1,1,1,1,1,4,5,1,6,1,1,1,10,16,16,6,1,1,6,1,5,5,1,1,1,1,2,1,6,1,6,16,1,1,1,10,3,2,1,1,1,1,2,4,6,9,2,4,9,9,1,4,1,5,10,1,10,1,1,1,4,1,1,1,6,4,6,1,2,1,1,1,1,1,6,1,16,1,1,1,1,1,1,1,1,1,1,10,1,1,1,1,1,1,10,1,1,10,1,1,1,5,1,1,10,10,10,1,1,10,1,1,1,6,16,1,1,2,1,1,1,1,1,1,1,1,1,1,5,4,1,1,1,10,15,6,1,1,1,2,1,16,1,4,2,4,2,2,14,9,1,1,2,2,1,1,1,16,16,1,2,1,1,1,3,1,1,9,1,10,10,1,2,2,4,1,2,15,3,16,1,1,6,1,10,3,1,16,1,1,1,4,1,1,1,2,1,2,1,1,1,1,1,15,1,2,1,1,4,1,10,4,3,3,1,1,2,3,5,2,1,16,1,1,1,1,10,1,1,1,1,1,6,1,1,2,1,2,10,1,1,1,1,6,10,3,1,1,1,1,1,10,1,10,2,2,2,10,10,1,15,1,6,3,2,1,16,6,2,7,1,1,10,10,1,1,5,1,1,10,5,1,2,2,10,1,10,7,1,2,1,1,16,1,10,1,10,1,1,1,16,10,1,6,10,1,10,1,5,1,1,2,1,10,16,1,3,2,6,2,2,3,16,10,6,1,2,2,2,1,9,1,2,1,5,2,8,1,1,10,16,3,1,1,6,1,16,5,9,1,1,1,1,1,1,9,1,10,3,1,10,14,1,5,1,1,1,1,1,16,4,2,16,1,1,1,1,10,1,1,15,1,1,1,9,1,1,10,1,16,10,6,10,3,1,1,1,1,1,1,1,1,1,10,1,1,1,1,10,2,1,1,1,
7,
7,
7,
8,
8,
9,
9,
9,
9,
9,
9,
9,
9,
9,
14,
14,
14,
14,
15,
15,
15,
15,
15,
7,
7,
7,
8,
8,
9,
9,
9,
9,
9,
9,
9,
9,
9,
14,
14,
14,
14,
15,
15,
15,
15,
15,
7,
7,
7,
8,
8,
9,
9,
9,
9,
9,
9,
9,
9,
9,
14,
14,
14,
14,
15,
15,
15,
15,
15,
])


dataset = pd.read_csv(url)
X=dataset

X=X.fillna(0)

# Apply SMOTE + ENN
sm = SMOTEENN()
X_resampled,y_resampled = sm.fit_sample(X, y)

print(X_resampled.shape)
#print(y_resampled.shape)
np.savetxt('smotex.csv',X_resampled, delimiter=',')
np.savetxt('smotey.csv',y_resampled, delimiter=',')
   




