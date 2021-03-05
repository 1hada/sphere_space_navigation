"""
This is a helper class to make a cspace with circular obstacles.
This class also shows the cspace via matplotlib.

author : Jonathan Sanabria
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import cv2


class MatplotlibCSpace():

  @staticmethod
  def view_3d_surface(X,Y,Z,UV=None,clip_lower=-1,clip_upper=10,lower_z=-1.01,upper_z=10.01,reduce_UV_by=250,gradient_title="Unit Gradient directions",potential_field_title="Potential Field"):
    # https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
    assert(X.shape == Y.shape)
    assert(X.shape == Z.shape)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    Z =  np.clip(Z, clip_lower, clip_upper)
                 
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    if not isinstance(UV,type(None)):
      ax2 = fig.add_subplot(122)
      W = np.zeros(Y.shape)
      def reduce(data):
        reduce_factor = reduce_UV_by
        return data[::reduce_factor,::reduce_factor]
      UV = UV/np.linalg.norm(UV,axis=0)
      ax2.quiver(reduce(X), reduce(Y), reduce(UV[0]), reduce(UV[1]),scale=1/0.025) 
      ax2.set_aspect('equal', adjustable='box')
      ax2.set_title(gradient_title)                       
           
    # Customize the z axis.
    ax.view_init(75, -72)
    ax.set_zlim(lower_z, upper_z)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(potential_field_title)                       
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
  @staticmethod
  def make_circle_obstacle(mesh_shape=(100,100),location=(0,0),radius=1,unit=10,height=8):
    obstacle = np.zeros(mesh_shape)
    x_location_fixed = int(unit*location[0]+mesh_shape[0]/2)
    y_location_fixed = int(unit*location[1]+mesh_shape[1]/2) 
    obstacle = cv2.circle(obstacle, (x_location_fixed,y_location_fixed),unit*radius,height, -1)  
    return obstacle
    
  @staticmethod
  def get_cspace(xy_limit=5,step=0.001):
    
    # Make data.
    X = np.arange(-xy_limit, xy_limit, step)
    Y = np.arange(-xy_limit, xy_limit, step)
    X, Y = np.meshgrid(X, Y)
    mesh_shape = X.shape
    cell_count = mesh_shape[0]
    unit = int(cell_count/(2*xy_limit))
    Z = np.zeros(X.shape) 
    
    return X,Y,Z,mesh_shape,unit
      

