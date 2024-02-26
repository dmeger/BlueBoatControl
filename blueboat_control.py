'''
This file holds a BlueBoat simulator using physics functions borrowed from a previous 
research project. Those are: 
Copyright (c) 2017, Juan Camilo Gamboa Higuera, Anqi Xu, Victor Barbaros, Alex Chatron-Michaud, David Meger
Copyright (c) 2024, Zhizun Wang, Louis Petit, David Meger

The GUI is new in 2020 and was started from the pendulum code of Wesley Fernandes
https://pastebin.com/zTZVi8Yv
python simple pendulum with pygame

The rest of the file and instructions are written by David Meger for the purposes of supporting
his teaching in RL and Robotics. Please use this freely for any purpose, but acknowledgement of sources
is always welcome.
'''
 
import pygame
import math
import numpy as np
from scipy.integrate import ode
import clock

# The very basic code you should know and interact with starts here. Sets some variables that you 
# might change or add to, then defines a function to do control that is currently empty. Add
# more logic in and around that function to make your controller work/learn!

x0 = [0,0,0,0,0,0]                      # This specifies the average starting state               
                                        # The meaning of the state dimensions are 
                                        # state[0] : boat position (x)
                                        # state[1] : boat position (y)
                                        # state[2] : boat angle (theta)
                                        # state[3] : boat velocity (x)
                                        # state[4] : boat velocity (y)
                                        # state[5] : boat angular velocity (theta_dot)

goal = np.array([ 0, 0, 0, np.pi ])     # This is where we want to end up. Perfectly at the centre
                                        # of the screen, with the boat pointing to the right.

# TODO: Fill in this function
def computeControl( x ):

    control = [1,0,0]

    return control

# After this is all the code to run the BlueBoat physics, draw it on the screen, etc. 
# You should not have to change anything below this, but are encouraged to read and understand
# as much as possible.

# VARIABLES FOR GUI/INTERACTION
screen_width, screen_height = 800, 800   # set the width and height of the window
                           # (you can increase or decrease if you want to, just remind to keep even numbers)
Done = False                # if True,out of while loop, and close pygame
Pause = False               # when True, freeze the boat. This is 
                            # for debugging purposes
 
#COLORS
white = (255,255,255)
black = (0,0,0)
gray = (150, 150, 150)
Dark_red = (150, 0, 0)
radius = 20
coord_to_screen_scaling = 100.0
boat_img_size = (100,49)
trailer_img_size = (300,125)
trailer_pos = (400,100)

#BEFORE STARTING GUI
pygame.init()
pygame.display.set_caption("BlueBoat Control") # set the title of the window
pygame.joystick.init() # initialize the joystick
joystick = pygame.joystick.Joystick(0) # create a joystick object
joystick.init() # initialize the joystick

background = pygame.display.set_mode((screen_width, screen_height))
#clock = pygame.time.Clock()
boat_img = pygame.transform.smoothscale( pygame.image.load("img/bb.png").convert_alpha(), boat_img_size)
trailer_img = pygame.transform.smoothscale( pygame.image.load("img/trailer.png").convert_alpha(), trailer_img_size)

# A simple class to simulate BlueBoat physics using an ODE solver
class BlueBoat(object):
 
    # State holds x, x_dot, theta_dot, theta (radians)
    def __init__(self, X0):  
        self.g = 9.82
        self.m = 0.5
        self.M = 0.5
        self.l = 0.5
        self.b = 1.0

        self.X0 = self.x = np.array(x0,dtype=np.float64).flatten()
        self.x = self.X0
        self.t = 0

        self.u = 0

        # This is a key line that makes this class an accurate version of BlueBoat dynamics.
        # The ODE solver is connected with our instantaneous dynamics equations so it can do 
        # the hard work of computing the motion over time for us.
        self.solver = ode(self.dynamics).set_integrator('dopri5', atol=1e-12, rtol=1e-12) 
        self.set_state(self.x)    

    # For internal use. This connects up the local state in the class
    # with the variables used by our ODE solver.
    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float64).flatten()
        self.solver = self.solver.set_initial_value(self.x)
        self.t = self.solver.t

    # Convenience function. Allows for quickly resetting back to the initial state to
    # get a clearer view of how the control works.
    def reset(self):
        self.x = self.X0
        self.t = 0
        self.set_state(self.x)

    TRAILER_LEFT_X = 0
    TRAILER_LEFT_Y = 0
    TRAILER_RIGHT_X = 4
    TRAILER_RIGHT_Y = 2
    TRAILER_HEIGHT = 0.2
    TRAILER_WIDTH = 0.1

    def to_screen(self,x,y):
        return (int(screen_width/2+x*coord_to_screen_scaling),int(screen_height/2+y*coord_to_screen_scaling))

    def blitRotate(self,surf, image, pos, originPos, angle):

        # offset from pivot to center
        image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
        
        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-angle)

        # roatetd image center
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

        # rotate and blit the image
        surf.blit(rotated_image, rotated_image_rect)

    # Draw the boat
    def draw(self, bg):  
        boat_rad = 20.0
        boat_centre = self.to_screen(self.x[0],self.x[1])
        
        #boat_direction = (int(boat_centre[0]+boat_rad*np.cos(self.x[2])),int(boat_centre[1]+boat_rad*np.sin(self.x[2])))
        #pygame.draw.circle(bg, Dark_red, boat_centre, radius - 2)
        #pygame.draw.lines(bg, black, False, [boat_centre, boat_direction], 2)

        self.blitRotate(bg,boat_img,boat_centre,(int(boat_img_size[0]/2),int(boat_img_size[1]/2)),self.x[2]*180.0/np.pi)

        boat_motor = (int(boat_centre[0]-boat_img_size[0]/2*np.cos(self.x[2])),int(boat_centre[1]+boat_img_size[0]/2*np.sin(self.x[2])))
        linear_thrust_arrow = (int(boat_motor[0]-self.u[0]*np.cos(self.x[2])),int(boat_motor[1]+self.u[0]*np.sin(self.x[2])))
        angular_thrust_arrow = (int(boat_motor[0]-5*self.u[1]*np.sin(self.x[2])),int(boat_motor[1]-5*self.u[1]*np.cos(self.x[2])))
        pygame.draw.lines(bg, Dark_red, False, [boat_motor, linear_thrust_arrow], 2)
        pygame.draw.lines(bg, Dark_red, False, [boat_motor, angular_thrust_arrow], 2)

        #pygame.draw.rect(bg,black,pygame.Rect(self.to_screen(self.TRAILER_LEFT_X,self.TRAILER_LEFT_Y),(self.TRAILER_HEIGHT*coord_to_screen_scaling,self.TRAILER_WIDTH*coord_to_screen_scaling)))

    def minangle(theta):
        while theta > np.pi:
            theta = theta - 2*np.pi
        while theta < -np.pi:
            theta = theta + 2*np.pi

    # These equations are simply typed in from the dynamics 
    # on the assignment document. They have been derived 
    # for a boat with a motor and a rudder, and are a simplified
    # version of the full dynamics of the boat.
    def dynamics(self,t,z):

        f = self.u                                                              # u[0] body-centric thrust
                                                                                # u[1] rotation force
        
        # Compute global linear force
        fx = self.u[0] * np.cos(z[2])
        fy = -self.u[0] * np.sin(z[2])

        dz = np.zeros((6,1))
        dz[0] = z[3]                                                            # x
        dz[1] = z[4]                                                            # y
        dz[2] = z[5]                                                            # theta
        dz[3] = fx - 3*z[3]                                                   # dx/dt
        dz[4] = fy - 3*z[4]                                                   # dy/dt
        dz[5] = f[1] - 3*z[5]                                                 # dtheta/dt

        return dz

    # Takes the command, u, and applies it to the system for dt seconds.
    # Note that the solver has already been connected to the dynamics
    # function in the constructor, so this function is effectively
    # evaluating the dynamics. The solver does this in an "intelligent" way
    # that is more accurate than dt * accel, rather it evaluates the dynamics
    # at several points and correctly integrates over time.
    def step(self,u,dt=None):

        self.u = u

        if dt is None:
            dt = 0.005
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def get_state(self):
        return self.x

# The next two are just helper functions for the display.
# Draw a grid behind the BlueBoat, and the trailer.
def grid():  
    for x in range(50, screen_width, 50):
        pygame.draw.lines(background, gray, False, [(x, 0), (x, screen_height)])
        for y in range(50, screen_height, 50):
            pygame.draw.lines(background, gray, False, [(0, y), (screen_width, y)])
    background.blit(trailer_img,trailer_pos)
 
# Clean up the screen and draw a fresh grid and the BlueBoat with its latest state coordinates
def redraw(): 
    background.fill(white)
    grid()
    boat.draw(background)
     # Draw a solid blue circle in the center
    #pygame.draw.circle(background, (0, 0, 255), (250, 250), 75)
    pygame.display.update()
        # Flip the display
    pygame.display.flip()

# Starting here is effectively the main function.
# It's a simple GUI drawing loop that calls to your code to compute the control, sets it to the 
# BlueBoat class and loops the GUI to show what happened.
boat = BlueBoat(x0) 
print(boat)
state = boat.get_state()
print(state)

LINACCEL = 4.0
JOY_MAX_LIN_ACCEL = 16.0
ROTACCEL = 2.0
JOY_MAX_ROT_ACCEL = 8.0
control = [0,0]

while not Done:
    #clock.tick(30)             # GUI refresh rate
                     
    for event in pygame.event.get():
    #if False:
        if event.type == pygame.QUIT:                    
            Done = True                                   
        if event.type == pygame.KEYDOWN:    # keyboard control
            if event.key == pygame.K_r:     # "r" key resets the simulator
                control = [0,0]
                boat.reset()
            if event.key == pygame.K_p:     # holding "p" key freezes time
                Pause = True
            if event.key == pygame.K_UP:
                control[0] = control[0]+LINACCEL
            if event.key == pygame.K_DOWN:
                control[0] = control[0]-LINACCEL
            if event.key == pygame.K_RIGHT:
                control[1] = control[1]-ROTACCEL
            if event.key == pygame.K_LEFT:
                control[1] = control[1]+ROTACCEL    
            if event.key == pygame.K_q:
                Done = True
        if event.type == pygame.KEYUP:      # releasing "p" makes us live again
            if event.key == pygame.K_p:
                Pause = False
        if event.type == pygame.JOYAXISMOTION:      # xbox joystick controller control
            if event.axis == 1:  # Left stick vertical axis = throttle
                control[0] = JOY_MAX_LIN_ACCEL * -joystick.get_axis(1)
            if event.axis == 2:  # Right stick horizontal axis = steering 
                control[1] = JOY_MAX_ROT_ACCEL * -joystick.get_axis(2)
            
    if not Pause:
        #print(control)
        #control = computeControl( state )  # This is the call to the code you write
        state = boat.step(control)
        #print(state)

    redraw()
 
pygame.quit()
