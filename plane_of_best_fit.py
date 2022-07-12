import numpy as np
import matplotlib.pyplot as plt
import copy

""" 
NOTE this is only a demo and does not include the following:
 - co-planer UV plane axis identification
 - outlier point detection
"""
sample = 1000
pts = np.zeros((sample,3))
rand_scale = np.random.rand(sample,2)*20

origin = np.random.rand(3)*10 # randomly selecting origin 

u = np.random.rand(3)*10
u -= origin
u/= np.linalg.norm(u)


v = np.random.rand(3)*10
v -= origin
v /= np.linalg.norm(v)

normal = np.cross(u,v)
normal /= np.linalg.norm(normal)

for i in range(sample):
    # randomly generating points that lie on a plane with a noise between [0,2] along the normal
    pts[i,:] = origin+rand_scale[i,0]*u + rand_scale[i,1]*v + np.random.rand()*2* normal



fig = plt.figure()
ax = fig.add_subplot(projection="3d")


# Calculating the linear regression of the plane 
d = np.ones(sample)
abc = (np.linalg.inv(pts.T@pts)@pts.T)@d



# Calculating the best fit 
xx,yy = np.meshgrid(range(round(min(pts[:,0]))-1,round(max(pts[:,0]))+1,1),range(round(min(pts[:,1]))-1,round(max(pts[:,1]))+1,1))

# Generating the plane 
z = (1- (abc[0]*xx+ abc[1]*yy))/abc[2]


# Plotting the points and plane of best fit
ax.scatter(pts[:,0],pts[:,1],pts[:,2],marker="o",color='red')# the og points
ax.plot_surface(xx,yy,z,alpha=0.5)

# Labeling the axis
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
