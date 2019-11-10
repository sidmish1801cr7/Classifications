import numpy as np
import decimal
import matplotlib.pyplot as plt
D = decimal.Decimal
N = 100
data = [D(str(item)) for item in np.random.random(N)]
plt.hist(np.asarray(data, dtype='float'), bins=10, normed=True)
plt.show()


