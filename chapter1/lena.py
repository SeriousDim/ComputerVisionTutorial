import matplotlib.pyplot as plt
import scipy.misc

# Вместо метода lena()
lena = scipy.misc.ascent()

plt.figure()

plt.subplot(111)
plt.imshow(lena)

plt.show()
