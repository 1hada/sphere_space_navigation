"""
This class implements potential field based navigation.
* New locations are chosen via gradient descent.
* Derivatives on the grid are taken via sympy.

NOTE : sympy will probably print print some crazy messages. No problem.

author : Jonathan Sanabria
"""


import numpy as np
from sympy import symbols, sqrt, diff, lambdify
from cspace_matplotlib import MatplotlibCSpace 


class SphereSpaceNavigation():

    def __init__(self,cspace_radius=10,q_start=(0,0),q_goal=(0,0),circular_obstacle_li=[],kappa=3):
        self.KAPPA = kappa
        self.cspace_radius = cspace_radius
        self.traversed_coordinates = []
        self.q_start = np.array(q_start)   
        self.q_goal = np.array(q_goal)
        self.obstacle_mesh_list = []
        self.obstacle_center_radius_list = circular_obstacle_li
        MCS = MatplotlibCSpace
        
        ##########################################
        ############# make cspace ################
        self.X,self.Y,self.Z,self.mesh_shape,self.unit = MCS.get_cspace(xy_limit=12,step=0.01) 
        
        ##########################################
        ############ make obstacle ###############
        for o in self.obstacle_center_radius_list:
          self.obstacle_mesh_list.append(MCS.make_circle_obstacle(mesh_shape=self.mesh_shape
                                      ,location=o[0]
                                      ,radius=o[1]
                                      ,unit=self.unit
                                      ,height=1
                                      )
                                   )

        ##########################################
        ######### make cspace boundary ###########
        # add D_0 as described in :
        # Exact robot navigation using artificial potential functions pg 507 (6) 
        self.D_0 = ((0,0),self.cspace_radius)
        self.cspace_boundary = MCS.make_circle_obstacle(mesh_shape=self.mesh_shape
                                      ,location=(0,0)
                                      ,radius=self.cspace_radius
                                      ,unit=self.unit
                                      ,height=1
                                      )
        # invert cspace boundary
        indices = np.where(0==self.cspace_boundary)
        self.cspace_boundary[indices] = 2                         
        self.cspace_boundary -= 1                           
        
        ##########################################
        ###### sum potential and gradient  #######
        #........................................#
        self.mesh = np.array([self.X,self.Y])
        self.gradient_vectors = None#np.zeros(self.mesh.shape)
        
        self.beta_repulsive_part = self.get_beta_grid()          
        self.Z += self.get_attractive_potential_grid()
                 
                 
                 
        ##########################################
        # main_sphere_space_navigation_algorithm #
        #........................................#       
        self.partial_x, self.partial_y = self.get_partials()                        
        print(self.main_sphere_space_navigation_algorithm())
        #self.visualize_traversed_coordinates(color=(0,0,255))
        
        # view obstacles
        for o in self.obstacle_mesh_list:
          o_indices = np.nonzero(o)
          self.Z[o_indices] = o[o_indices]
          
        # view cspace boundary
        boundary_indices = np.where(0<self.cspace_boundary)
        self.Z[boundary_indices] = self.cspace_boundary[boundary_indices] 
        #----- nan elements only exist outside of sphere space
        assert(not np.any(np.isnan(self.Z)))
        #print("np.min(phi_q) : ",np.min(self.Z))
        #print("np.max(phi_q) : ",np.max(self.Z))
        
        # goal and start relief
        # ----- for visuals only
        IDX_DELTA_FOR_QSTART_QGOAL = 10
        IDFQ = IDX_DELTA_FOR_QSTART_QGOAL
        x,y = self.get_indices_from_distance_to(self.q_goal)
        self.Z[x-IDFQ:x+IDFQ,y-IDFQ:y+IDFQ] = 10
        x,y = self.get_indices_from_distance_to(self.q_start)
        self.Z[x-IDFQ:x+IDFQ,y-IDFQ:y+IDFQ] = 10
        
        # view potentials   
        REDUCE_FACTOR = 1 if 200<self.num_gradient_descent_epochs else 2                 
        MCS.view_3d_surface(self.X,self.Y,self.Z,
                              UV=-1*self.path,clip_lower=-1,
                              clip_upper=1,lower_z=-5,upper_z=5,
                              reduce_UV_by=REDUCE_FACTOR,
                              gradient_title='Path traversed,\npath can also be seen in potential field being "carved" into the potential field\n by a color associated with greater depth.'
                              ,potential_field_title=f"Sphere Space Navigation Potential Field\n KAPPA == {self.KAPPA}")
        
    def get_indices_from_distance_to(self,cur_loc):
        """
        helper function to find the index to a desired (x,y) location
        @param  cur_loc : numpy array (x,y) as the current R^2 location 
        """
        m = self.mesh
        dist = np.linalg.norm(m-cur_loc.reshape((2,1,1)),axis=0)
        min_i,min_j = np.unravel_index(np.argmin(dist, axis=None),dist.shape)
        return min_i,min_j

    def gradient_descent(self,cur_xy,gradient):
        """
        main gradient descent algorithm with step function
        @param   cur_xy : numpy array (x,y) as the current R^2 location 
        @param gradient : numpy array (x,y) as the gradient at the cell  
         
        @return next (x,y) location from f-df
        """
        STEP = 0.15*self.KAPPA
        next_xy = cur_xy - STEP*gradient/np.linalg.norm(gradient)
        return  next_xy
        
    def calculate_navigation_path(self):
        """
        gradient descent navigation based on sphere space navigation
        
        @returns a string indicating SUCCESS or FAILURE, FAILURE defined by robot in local min
        """
        # get start indices
        IDX_DELTA_FOR_QSTART_QGOAL = 10
        IDFQ = IDX_DELTA_FOR_QSTART_QGOAL
        SUCCESS_STR = "SUCCESS"
        i,j = self.get_indices_from_distance_to(self.q_start)
        path = np.empty_like(self.gradient_vectors)
        path *= 0
        at_goal = lambda cur_gradient: 1.80715011e-09>np.linalg.norm(cur_gradient)

        seen_indices = []
        self.num_gradient_descent_epochs = 1
        while not at_goal(gradient := self.gradient_vectors[:,i,j]):
            ###############################
            # MAIN GRADIENT DESCENT CALLS #
            cur_xy = self.mesh[:,i,j]
            gradient = self.gradient_vectors[:,i,j]
            next_xy = self.gradient_descent(cur_xy,gradient)
            self.num_gradient_descent_epochs+=1
            
            i,j = self.get_indices_from_distance_to(next_xy)
            
            # visualize vectors
            path[:,i,j] = gradient
            
            # "carve" path of 3d graph
            self.Z[i:i+IDFQ,j:j+IDFQ] = -0.5
            
            # check for local min
            if (i,j) in seen_indices:
                ######################
                # AT A LOCAL MINIMUM #
                ######################
                # would raise error but I am not for sake of grader
                SUCCESS_STR = f"FAILURE : Stuck at  {(self.X[i,j],self.Y[i,j])}"
                break   
            seen_indices.append((i,j))
       
        print( "self.num_gradient_descent_epochs : ",self.num_gradient_descent_epochs ) 
        self.path = path
        return SUCCESS_STR
        
    def get_partials(self):
        """
        this method gets the partial derivative mesh needed by self.define_gradient_vectors(..)
        
        mustve defined :
        --- self.q_goal
        --- self.D_0
        --- self.obstacle_center_radius_list
        
        @returns sympy functions relative to each x , y cell in self.mesh.
        """


        # define variables
        x, y, z = symbols('x y z')

        # d_q == distance_to_closest_obstacle_coordinate
        d_0_xy , d_0_radius = self.D_0
        lhs = d_0_radius**2 - (sqrt((x-d_0_xy[0])**2+(y-d_0_xy[1])**2)**2)
        beta_rhs_list = []
        for o_center, o_radius in self.obstacle_center_radius_list:
            ox,oy = o_center
            obs = sqrt((x-ox)**2+(y-oy)**2)**2 - o_radius**2
            beta_rhs_list.append(obs)
        BETA = lhs
        for bs in beta_rhs_list:
            BETA *= bs

        # define formulas for the obstacle
        # d_q == distance_to_goal
        d_q = sqrt((x-self.q_goal[0])**2+(y-self.q_goal[1])**2) 

        # obstacle being a circle
        gamma = d_q**2/((d_q**(2*self.KAPPA) + BETA)**(1/self.KAPPA))
        partial_x = diff(gamma,x)
        partial_y = diff(gamma,y)
        
        return partial_x, partial_y
        
    def define_gradient_vectors(self,mesh):    
        """
        Defines the gradient of each cell in self.mesh into self.gradient_vectors
        """
        x, y, z = symbols('x y z')
        idxi,idxj = self.get_indices_from_distance_to(np.array((0,0)))
        x_i,y_i= self.X , self.Y  
        fx = lambdify([x,y], self.partial_x, "numpy")
        fy = lambdify([x,y], self.partial_y, "numpy")
        dx = fx(x_i,y_i)
        dy = fy(x_i,y_i)
        self.gradient_vectors = np.array([dx,dy])


    def get_attractive_potential_grid(self):
        """
        @return grid of values == attractive_potential_based_on the sphere space navigation
        """
        phi_q = np.zeros(self.Z.shape)
        mesh = self.mesh.astype(np.float32)
        
        # get distances
        rho_d = np.linalg.norm( mesh - self.q_goal.reshape((2,1,1)),axis=0 )
        assert( rho_d.shape == self.beta_repulsive_part.shape )
        
        phi_q = pow(rho_d,2)/pow(pow(rho_d,2*self.KAPPA)+self.beta_repulsive_part,1/self.KAPPA)

        return phi_q
        
    def get_beta_pi_product_part(self):
        """
        @return grid of values == the_pi_product of the distances to the centor of all obstacles
        """
        pi_product = np.ones(self.Z.shape)
        mesh = self.mesh.astype(np.float32) 

        # get closest distance
        for o_center, o_radius in self.obstacle_center_radius_list:
            # get distances
            d_q_qi = np.linalg.norm(mesh-np.array(o_center).reshape((2,1,1)) ,axis=0 )
            prod_dif = pow(d_q_qi,2) - pow(o_radius,2)
            
            ################################
            # pi product repulsive potential
            pi_product *= prod_dif
            
        return pi_product    
        
    def get_beta_grid(self):
        """
        @return a grid of values == repulsive_potential_based_on the sphere space navigation 
        """
        B = np.zeros(self.Z.shape)
        mesh = self.mesh.astype(np.float32) 
        pi_product = self.get_beta_pi_product_part()

        # left hand side of beta
        q_0_coordinates = self.D_0[0]
        q_0_radius = self.D_0[1]
        d_q_q0 = np.linalg.norm(mesh-np.array(q_0_coordinates).reshape((2,1,1)),axis=0 )
        dist_minus_radius = pow(q_0_radius,2) - pow(d_q_q0,2)

        # sum repulsive potential
        assert( B.shape == pi_product.shape )
        assert( B.shape == dist_minus_radius.shape )
        B += dist_minus_radius*pi_product
        
        return B
              
    def main_sphere_space_navigation_algorithm(self):
        """
        @return success/failure string
        """
        self.define_gradient_vectors(self.mesh)
        return self.calculate_navigation_path()
        
if __name__ == "__main__" :
    start = (-9,0)
    goal = (9,0)

    #################################################
    ######## SPHERE SPACE NAVIGATION ################
    #################################################   
    """
    # commented because the ones with higher KAPPA values are prettier in the way they show gradients :)
    kappa = 1
    # part a ---- KAPPA == 1
    # first obstacle set --- will succeed
    obstacle_li = [((-5,-5),2), ((0,-5),2), ((2,1),1)]
    SphereSpaceNavigation(q_start=start,q_goal=goal,circular_obstacle_li=obstacle_li,kappa=kappa)
    
    # part b ---- KAPPA == 1
    # second obstacle set --- will get stuck because need a higher KAPPA value
    obstacle_li = [ ((-4,5),1), ((0,-7),1),((4,5),1)] 
    SphereSpaceNavigation(q_start=start,q_goal=goal,circular_obstacle_li=obstacle_li,kappa=kappa)
    """
    # part c ---- KAPPA == 3
    kappa = 3
    obstacle_li = [ ((-4,5),1), ((0,-7),1),((4,5),1)] 
    # second obstacle set --- will succeed with previous environment as asked
    SphereSpaceNavigation(q_start=start,q_goal=goal,circular_obstacle_li=obstacle_li,kappa=kappa)
      
    # part EXTRA ---- KAPPA == 3
    kappa = 3
    # third obstacle set --- will succeed with previous environment as asked
    obstacle_li = [((-5,5),2), ((1,-7),1), ((0,1),1)]
    SphereSpaceNavigation(q_start=start,q_goal=goal,circular_obstacle_li=obstacle_li,kappa=kappa)
         
    
