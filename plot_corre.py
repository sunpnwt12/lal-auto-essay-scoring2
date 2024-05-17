import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tables = {
    'exp': ['exp046', 'exp051', 'exp052', 'exp053', 'exp054', 'exp055'],
    'cv': [0.7820, 0.7805, 0.7823, 0.787, 0.7805, 0.7815],
    'lb': [0.768, 0.772, 0.777, 0.774, 0.796, 0.770],
}

record = pd.DataFrame(tables)

sns.scatterplot(record, x='cv', y='lb')
for idx, txt in enumerate(record['exp']):
    plt.text(record['cv'][idx], record['lb'][idx], txt)

plt.grid()
plt.show()
    