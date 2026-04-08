import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.show()

data = np.random.randn(5,2,1000)

plt.hist(data.reshape(-1), bins=30)
plt.show()

labels = ['програмування', 'рибалка', 'лежати', 'лежати на дивані']
sizes = [20,10,50,20]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.show()

apple = np.random.normal(150,20,100)
banana = np.random.normal(120,15,100)
orange = np.random.normal(180, 25, 100)
pear = np.random.normal(160, 18, 100)

data = [apple, banana, orange, pear]

labels = ['Apple', 'Banana', 'Orange', 'Pear']

plt.boxplot(data, labels=labels)

plt.grid(True)

plt.show()

