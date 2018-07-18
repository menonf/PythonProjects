import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas





patches=[]
fig, ax = plt.subplots()
sf = shp.Reader("Districts")
plt.figure(figsize=(15, 15))
for shape in sf.shapeRecords():
    x = [i[0]  for i in shape.shape.points[:]]
    y = [i[1]  for i in shape.shape.points[:]]
    plt.plot(x,y, 'K', linewidth = 0.5)
    #polygon = Polygon(np.array([x, y]).T, closed=True)  # get one single polygon
   # patches.append(polygon)  # add it to a final array


# webData = pandas.DataFrame.from_csv("bulk.csv", header=0).set_index('geography code')
# demographic = webData.iloc[:, 2:3]
#
# pop_density = np.array(map(float,demographic)) #convert to float
#
# p = PatchCollection(patches, cmap="Blues")
# p.set_array(pop_density)
# ax.add_collection(p)
plt.savefig('lot.jpg')
plt.show()



