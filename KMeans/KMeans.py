import math
import matplotlib.pyplot as plt
coordinates=[(3,7),(5,4),(4,3),(4,8),(6,3),(3,8)]

centroid1=[3,7]
centroid2=[5,4]
cluster1=[]
cluster2=[]
maxDis1=-1;
maxDis2=-1;
cluster1.append(tuple(centroid1))
cluster2.append(tuple(centroid2));
for points in coordinates:
   distance1=math.sqrt( pow(points[0]-centroid1[0],2) + pow(points[1]-centroid1[1],2))
   distance2 = math.sqrt(pow(points[0] - centroid2[0], 2) + pow(points[1] - centroid2[1], 2))
   if distance1 !=0 and distance2 !=0:
       if distance1 < distance2:
          centroid1[0]=(points[0]+centroid1[0])/2
          centroid1[1] = (points[1] + centroid1[1]) / 2
          cluster1.append(points)
          if maxDis1<distance1:
             maxDis1=distance1

       else:

          centroid2[0] = (points[0] + centroid2[0]) / 2
          centroid2[1] = (points[1] + centroid2[1]) / 2
          cluster2.append(points)
          if maxDis2<distance2:
             maxDis2=distance2

circle1 = plt.Circle((centroid1[0],centroid1[1]), maxDis1, color='red',fill=False)
fig, ax = plt.subplots()
ax.add_patch(circle1)
circle2 = plt.Circle((centroid2[0],centroid2[1]), maxDis2, color='green',fill=False)
ax.add_patch(circle2)
ax.legend([circle1,circle2], ['Cluster1','Cluster2'])

print("\nCluster 1 points are:")
for points in cluster1:
   print(points)
   plt.scatter(points[0],points[1],color = 'red')

print("\nCluster 2 points are:")
for points in cluster2:
      print(points)
      plt.scatter(points[0], points[1],color = 'green')

plt.show();
