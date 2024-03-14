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

import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np
from scipy.integrate import ode
import clock
from utils import minangle

# The very basic code you should know and interact with starts here. Sets some variables that you 
# might change or add to, then defines a function to do control that is currently empty. Add
# more logic in and around that function to make your controller work/learn!

# x0 = [0,0,0,0,0,0]                      # This specifies the average starting state               
                                        # The meaning of the state dimensions are 
                                        # state[0] : boat position (x)
                                        # state[1] : boat position (y)
                                        # state[2] : boat angle (theta)
                                        # state[3] : boat velocity (x)
                                        # state[4] : boat velocity (y)
                                        # state[5] : boat angular velocity (theta_dot)

# goal = np.array([ 0, 0, 0, np.pi ])     # This is where we want to end up. Perfectly at the centre
                                        # of the screen, with the boat pointing to the right.

# reward model for blue boat
class BlueBoatReward(object):
    def __init__(self, boat_pos, trailer_pos):  
        super(BlueBoatReward, self).__init__()
        self.boat_pos = np.asarray(boat_pos)
        self.trailer_pos = np.asarray(trailer_pos)
        
    def reward(self):
        diff = self.boat_pos - self.trailer_pos
        x = np.linalg.norm(diff, ord=2)
        # reward = e^(-x)
        reward = np.exp(-x)
        return reward

# A simple class to simulate BlueBoat physics using an ODE solver
class BlueBoat(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}
 
    # State holds x, x_dot, theta_dot, theta (radians)
    def __init__(self, model=None, X0=None):
        super(BlueBoat, self).__init__()
        self.g = 9.82
        self.m = 0.5
        self.M = 0.5
        self.l = 0.5
        self.b = 1.0

        self.X0 = np.array(X0,dtype=np.float32).flatten()
        self.x = self.X0
        self.t = 0.0

        self.u = np.array([0.0, 0.0], dtype=np.float32)
        
        # This is a key line that makes this class an accurate version of BlueBoat dynamics.
        # The ODE solver is connected with our instantaneous dynamics equations so it can do 
        # the hard work of computing the motion over time for us.
        self.solver = ode(self.dynamics).set_integrator('dopri5', atol=1e-12, rtol=1e-12) 
        self.set_state(self.x)   
        
        # self.x0 = np.array([0,0,0,0,0,0], dtype=np.float32)
        # self.goal = np.array([ 0, 0, 0, np.pi ], dtype=np.float32)

        self.Done = False                # if True,out of while loop, and close pygame
        self.Pause = False               # when True, freeze the boat. This is for debugging
        

        # COLORS
        self.white = (255,255,255)
        self.black = (0,0,0)
        self.gray = (150, 150, 150)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_red = (150, 0, 0)
        self.light_blue = (173, 216, 230)
        self.light_red = (255, 182, 193)
        self.light_green = (144, 238, 144)
        self.dark_green = (0, 150, 0)
        self.light_grey = (150, 150, 150)

        # DISPLAY
        self.screen_width = 800
        self.screen_height = 800
        self.screen_center = (self.screen_width // 2, self.screen_height // 2)
        self.coord_to_screen_scaling = 100.0
        self.boat_img_size = (100,49)

        # TRAILER DISPLAY AND GEOMETRY
        self.trailer_img_size = (300*1.15,125*1.15)
        self.trailer_approach_dist = self.trailer_img_size[0] * 0.2
        self.trailer_threshold = (self.trailer_img_size[0] * 0.06, self.trailer_img_size[1] * 0.08)
        self.trailer_pos = (400,200) # It's in pixels!
        self.trailer_yaw = np.pi / 4
        self.trailer_inside_prop = (0.30, 0.525)
        self.trailer_centre = (self.trailer_pos[0] + self.trailer_img_size[0] * self.trailer_inside_prop[0], 
                               self.trailer_pos[1] + self.trailer_img_size[1] * self.trailer_inside_prop[1])
        self.trailer_approach_pt_1 = (int(self.trailer_centre[0] - self.trailer_approach_dist * np.cos(self.trailer_yaw)), int(self.trailer_centre[1] + self.trailer_approach_dist * np.sin(self.trailer_yaw)))
        self.trailer_approach_pt_0 = (int(self.trailer_centre[0] - 2 * self.trailer_approach_dist * np.cos(self.trailer_yaw)), int(self.trailer_centre[1] + 2 * self.trailer_approach_dist * np.sin(self.trailer_yaw)))
                                      
        # DRIVING MODES DISPLAY
        self.throttle_bar_width = 200
        self.throttle_bar_height = 20
        self.throttle_bar_position = (self.screen_width // 2 - self.throttle_bar_width // 2, 700)
        self.steering_bar_width = 200
        self.steering_bar_height = 20
        self.steering_bar_position = (self.screen_width // 2 - self.steering_bar_width // 2, 750)
        self.driving_mode_position = (self.screen_width // 2 - self.steering_bar_width // 2, 650)

        # VELOCITY PROFILES DISPLAY
        self.vel_prof_height = 150
        self.vel_prof_width = 300
        self.lin_vel_prof_pos = (50, 0)
        self.ang_vel_prof_pos = (50, self.vel_prof_height)

        # FONT
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)

        # KEYBOARD CONTROL
        self.LINACCEL_INCR = 4.0
        self.ROTACCEL_INCR = 2.0

        # CONTROL
        self.FORWARD_MIN_LIN_ACCEL = 10.0
        self.FORWARD_MAX_LIN_ACCEL = 30.0
        self.REVERSE_MIN_LIN_ACCEL = 2.0
        self.REVERSE_MAX_LIN_ACCEL = 8.0
        self.MAX_ROT_ACCEL = 4.0
        self.THROTTLE_THRESHOLD = 2.0
        self.control = np.array([0.0,0.0], dtype=np.float32)


        pygame.init()
        pygame.display.set_caption("BlueBoat Control 2") # set the title of the window  
        self.background = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.boat_img = pygame.transform.smoothscale( pygame.image.load("img/bb.png").convert_alpha(), self.boat_img_size)
        self.trailer_img = pygame.transform.smoothscale( pygame.image.load("img/trailer.png").convert_alpha(), self.trailer_img_size)
        self.model = model

        self.bpos = self.x[:2]
        self.bpos = np.asarray(self.bpos, dtype=np.float32)

        from_trailer_pos = self.from_screen(self.trailer_centre[0], self.trailer_centre[1])
        self.tpos = np.asarray(from_trailer_pos, dtype=np.float32)
        self.initial_dist = np.linalg.norm((self.bpos-self.tpos), ord=2)
        
        high = np.array([self.FORWARD_MAX_LIN_ACCEL, self.MAX_ROT_ACCEL], dtype=np.float32)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)     

        # observation space is the combination of state, action, and trailer position 
        size_meters = self.from_screen(self.screen_width, self.screen_height)
        self.grid_shape = np.asarray(size_meters, dtype=np.float32)
        self.high_s = np.array([self.grid_shape[0], self.grid_shape[1], np.pi, 10, 10, np.pi], dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(-self.high_s, self.high_s, shape=(6, ), dtype=np.float32),
            "action": spaces.Box(-high, high, shape=(2, ), dtype=np.float32),
            "target": spaces.Box(-self.grid_shape, self.grid_shape, shape=(2, ), dtype=np.float32),
            })
         
    # For internal use. This connects up the local state in the class
    # with the variables used by our ODE solver.
    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float32).flatten()
        self.solver = self.solver.set_initial_value(self.x)
        self.t = self.solver.t
        
    def get_state(self):
        return self.x
    
    def get_obs(self):
        return {"state": self.x, "action": self.u, "target": self.tpos}
    
    def get_reward(self):
        self.bpos = self.x[:2]
        self.bpos = np.asarray(self.bpos)
        reward = 0.0
        threshold = 0.1 * self.initial_dist
        boundary = 2.0 * self.initial_dist
        diff = self.bpos - self.tpos
        curr_dist = np.linalg.norm(diff, ord=2)
        
        # print(self.initial_dist)
        # print(curr_dist)
        if self.is_in_trailer_with_yaw(self.bpos[0], self.bpos[1], self.x[2]):
            reward = 10.0
        elif self.is_in_trailer(self.bpos[0], self.bpos[1]):
            reward = 1.0
        # elif abs(curr_dist) >= boundary:
        #     reward = -10.0
        else:
            # if self.initial_dist <= curr_dist:
            #     reward = -0.01
            # else:
                # prevent reward from zero division
                reward = 1.0 / (curr_dist + 1.0)
                # reward = 0.01
        return reward

    # Convenience function. Allows for quickly resetting back to the initial state to
    # get a clearer view of how the control works.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.x = self.X0
        self.t = 0.0
        self.set_state(self.x)
        
        obs = self.get_obs()
        state = self.get_state()
        reward = self.get_reward()
        info = {"t": self.t, "action": self.u}
        # return obs, state, reward, info
        
        # print(obs)
        # print(info)
        
        return obs, info

    def to_screen(self,x,y):
        return (int(self.screen_width/2+x*self.coord_to_screen_scaling),
                int(self.screen_height/2+y*self.coord_to_screen_scaling))
    
    def from_screen(self,x,y):
        return ((x-self.screen_width/2)/self.coord_to_screen_scaling,
                (y-self.screen_height/2)/self.coord_to_screen_scaling)
    
    def is_inside_map(self,x,y):
        boat_centre2 = self.to_screen(x,y)
        return boat_centre2[0] > 0 and boat_centre2[0] < self.screen_width and boat_centre2[1] > 0 and boat_centre2[1] < self.screen_height
    
    def is_in_trailer(self,x,y):
        boat_centre = self.to_screen(x,y)
        # Rotate boat coordinates relative to the trailer's yaw angle
        rotated_x = (boat_centre[0] - self.trailer_centre[0]) * np.cos(-self.trailer_yaw) - (boat_centre[1] - self.trailer_centre[1]) * np.sin(-self.trailer_yaw)
        rotated_y = (boat_centre[0] - self.trailer_centre[0]) * np.sin(-self.trailer_yaw) + (boat_centre[1] - self.trailer_centre[1]) * np.cos(-self.trailer_yaw)
        # Check if the rotated boat coordinates are within the specified threshold around the trailer
        return rotated_x > -self.trailer_threshold[0] and rotated_x < self.trailer_threshold[0] and rotated_y > -self.trailer_threshold[1] and rotated_y < self.trailer_threshold[1]

    def is_in_trailer_with_yaw(self,x,y,theta):
        theta_threshold = np.pi / 18 # 10 degrees
        return self.is_in_trailer(x, y) and abs(minangle(theta - self.trailer_yaw)) < theta_threshold

    def blitRotate(self, surf, image, pos, originPos, angle):

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
    def draw(self, bg, boat_img, trailer_img):  
        boat_rad = 20.0
        boat_centre = self.to_screen(self.x[0],self.x[1])

        self.blitRotate(bg,boat_img,boat_centre,(int(self.boat_img_size[0]/2),
                                                 int(self.boat_img_size[1]/2)),self.x[2]*180.0/np.pi)

        boat_motor = (int(boat_centre[0]-self.boat_img_size[0]/2*np.cos(self.x[2])),
                      int(boat_centre[1]+self.boat_img_size[0]/2*np.sin(self.x[2])))
        linear_thrust_arrow = (int(boat_motor[0]-self.u[0]*np.cos(self.x[2])),
                               int(boat_motor[1]+self.u[0]*np.sin(self.x[2])))
        angular_thrust_arrow = (int(boat_motor[0]-5*self.u[1]*np.sin(self.x[2])),
                                int(boat_motor[1]-5*self.u[1]*np.cos(self.x[2])))
        pygame.draw.lines(bg, self.dark_red, False, [boat_motor, linear_thrust_arrow], 2)
        pygame.draw.lines(bg, self.dark_red, False, [boat_motor, angular_thrust_arrow], 2)

        # draw boat centre
        pygame.draw.circle(bg, self.blue, boat_centre, 5)

    def draw_throttle_bar(self, bg, throttle, clamped_throttle):
        throttle_percentage = (throttle / self.FORWARD_MAX_LIN_ACCEL) * 100
        clamped_throttle_percentage = (clamped_throttle / self.FORWARD_MAX_LIN_ACCEL) * 100
        tick_color = self.black
        throttle_label = self.font.render("Throttle", True, self.black)
        if throttle >= 0:
            throttle_bar_color = self.light_green # green for forward
            pygame.draw.rect(bg, throttle_bar_color, (self.throttle_bar_position[0], self.throttle_bar_position[1], self.throttle_bar_width * throttle_percentage / 100, self.throttle_bar_height))
            bg.blit(throttle_label, (self.throttle_bar_position[0] - 100, self.throttle_bar_position[1]))
        else:
            throttle_bar_color = self.light_red # red for reverse
            pygame.draw.rect(bg, throttle_bar_color, (self.throttle_bar_position[0] +  self.throttle_bar_width * throttle_percentage / 100, self.throttle_bar_position[1], - self.throttle_bar_width * throttle_percentage / 100, self.throttle_bar_height))
            bg.blit(throttle_label, (self.throttle_bar_position[0] + 10, self.throttle_bar_position[1]))
        # Draw the tick as a vertical line
        pygame.draw.lines(bg, tick_color, False, [(self.throttle_bar_position[0] + self.throttle_bar_width * clamped_throttle_percentage / 100, self.throttle_bar_position[1]), (self.throttle_bar_position[0] + self.throttle_bar_width * clamped_throttle_percentage / 100, self.throttle_bar_position[1] + self.throttle_bar_height)], 2)

    def draw_steering_bar(self, bg, steering, clamped_steering):
        steering_percentage = (-steering / self.MAX_ROT_ACCEL) * 100
        clamped_steering_percentage = (-clamped_steering / self.MAX_ROT_ACCEL) * 100
        steering_bar_color = self.light_blue
        tick_color = self.black
        steering_label = self.font.render("Steering", True, self.black)
        if steering <= 0:
            pygame.draw.rect(bg, steering_bar_color, (self.steering_bar_position[0], self.steering_bar_position[1], self.steering_bar_width * steering_percentage / 100, self.steering_bar_height))
            bg.blit(steering_label, (self.steering_bar_position[0] - 100, self.steering_bar_position[1]))
        else:
            pygame.draw.rect(bg, steering_bar_color, (self.steering_bar_position[0] + self.steering_bar_width * steering_percentage / 100, self.steering_bar_position[1], - self.steering_bar_width * steering_percentage / 100, self.steering_bar_height))
            bg.blit(steering_label, (self.steering_bar_position[0] + 10, self.steering_bar_position[1]))
        # Draw the tick as a vertical line
        pygame.draw.lines(bg, tick_color, False, [(self.steering_bar_position[0] + self.steering_bar_width * clamped_steering_percentage / 100, self.steering_bar_position[1]), (self.steering_bar_position[0] + self.steering_bar_width * clamped_steering_percentage / 100, self.steering_bar_position[1] + self.steering_bar_height)], 2)
    
    # draw_velocity_profiles TODO
        
    # add_to_velocity_profiles TODO

    # display_driving_mode TODO
        
    # display_path_history TODO
        
    # add_to_path_history TODO

    def display_inside_map_info(self, bg):
        # display if the boat is inside the map or not
        if self.is_inside_map(self.x[0], self.x[1]):
            pygame.draw.rect(bg, self.light_green, (0, 0, 20, 20))
        else:
            pygame.draw.rect(bg, self.light_red, (0, 0, 20, 20))
            outside_map_text = self.font.render("Outside Map", True, self.black)
            bg.blit(outside_map_text, (25, 0))

    def display_inside_trailer_info(self, bg):
        # display if the boat is inside the trailer or not
        rotated_rect = pygame.Surface((self.trailer_threshold[0]*2, self.trailer_threshold[1]*2), pygame.SRCALPHA) 

        if self.is_in_trailer_with_yaw(self.x[0], self.x[1], self.x[2]):
            pygame.draw.rect(rotated_rect, self.light_green, (0, 0, self.trailer_threshold[0]*2, self.trailer_threshold[1]*2), 0)
            rotated_rect = pygame.transform.rotate(rotated_rect, self.trailer_yaw * 180.0 / np.pi) # Rotate the surface
            bg.blit(rotated_rect, rotated_rect.get_rect(center = self.trailer_centre)) # Blit the rotated surface to the screen
        else:
            pygame.draw.rect(rotated_rect, self.light_red, (0, 0, self.trailer_threshold[0]*2, self.trailer_threshold[1]*2), 0)
            rotated_rect = pygame.transform.rotate(rotated_rect, self.trailer_yaw * 180.0 / np.pi) # Rotate the surface
            bg.blit(rotated_rect, rotated_rect.get_rect(center = self.trailer_centre)) # Blit the rotated surface to the screen

        if self.is_in_trailer(self.x[0], self.x[1]):
            pygame.draw.circle(bg, self.dark_green, self.trailer_centre, 5)
        else:
            pygame.draw.circle(bg, self.dark_red, self.trailer_centre, 5)

    def display_approach_points(self, bg, TFS):
        if TFS == 0:
            pygame.draw.circle(bg, self.light_red, self.trailer_approach_pt_0, 5)
            pygame.draw.circle(bg, self.light_red, self.trailer_approach_pt_1, 5)
        elif TFS == 1:
            pygame.draw.circle(bg, self.light_green, self.trailer_approach_pt_0, 5)
            pygame.draw.circle(bg, self.light_red, self.trailer_approach_pt_1, 5)
        elif TFS == 2:
            pygame.draw.circle(bg, self.light_green, self.trailer_approach_pt_0, 5)
            pygame.draw.circle(bg, self.light_green, self.trailer_approach_pt_1, 5)
        else:
            pygame.draw.circle(bg, self.light_grey, self.trailer_approach_pt_0, 5)
            pygame.draw.circle(bg, self.light_grey, self.trailer_approach_pt_1, 5)

    # The next two are just helper functions for the display.
    # Draw a grid behind the BlueBoat, and the trailer.
    def grid(self, background, boat_img, trailer_img):  
        for x in range(50, self.screen_width, 50):
            pygame.draw.lines(background, self.gray, False, [(x, 0), (x, self.screen_height)])
            for y in range(50, self.screen_height, 50):
                pygame.draw.lines(background, self.gray, False, [(0, y), (self.screen_width, y)])
        # display the trailer_img with a rotation equal to trailer_yaw around trailer_centre
        self.blitRotate(background, trailer_img, self.trailer_centre, (int(self.trailer_img_size[0] * self.trailer_inside_prop[0]), int(self.trailer_img_size[1] * self.trailer_inside_prop[1])), self.trailer_yaw * 180.0 / np.pi)

    # Update the trailer position and orientation randomly within the screen
    def update_trailer_position(self):
        image_threshold = 150
        self.trailer_pos = (np.random.randint(image_threshold, self.screen_width - image_threshold), 
                       np.random.randint(image_threshold, self.screen_height - image_threshold))
        self.trailer_yaw = np.random.uniform(-np.pi, np.pi)
        self.trailer_centre = (self.trailer_pos[0] + self.trailer_img_size[0] * self.trailer_inside_prop[0], 
                          self.trailer_pos[1] + self.trailer_img_size[1] * self.trailer_inside_prop[1])
        self.trailer_approach_pt_1 = ((self.trailer_centre[0] - self.trailer_approach_dist * np.cos(self.trailer_yaw)), 
                                (self.trailer_centre[1] + self.trailer_approach_dist * np.sin(self.trailer_yaw)))
        self.trailer_approach_pt_0 = ((self.trailer_centre[0] - 2 * self.trailer_approach_dist * np.cos(self.trailer_yaw)), 
                                (self.trailer_centre[1] + 2 * self.trailer_approach_dist * np.sin(self.trailer_yaw)))
        
    # Clean up the screen and draw a fresh grid and the BlueBoat with its latest state coordinates
    def redraw(self, background, boat_img, trailer_img): 
        background.fill(self.white)
        self.grid(background, boat_img, trailer_img) # draw the grid
        self.display_inside_trailer_info(background) # display if the boat is inside the trailer or not
        # TODO: add if in auto TF mode: self.display_approach_points(background, 3) # display the approach points
        # display reference path here if needed
        self.draw(background, boat_img, trailer_img) # draw the boat 
        self.draw_throttle_bar(background, self.u[0], self.u[0]) # display throttle bar
        self.draw_steering_bar(background, self.u[1], self.u[1]) # display steering bar
        # TODO display driving mode
        self.display_inside_map_info(background) # display if the boat is inside the map or not
        self.display_reward(background) # display reward
        # TODO display velocity profiles if needed
        # TODO display time
        pygame.display.update()
        pygame.display.flip()

    def display_reward(self, bg):
        reward = self.get_reward()
        reward_text = self.font.render("Reward: " + str(reward), True, self.black)
        bg.blit(reward_text, (self.screen_width - 200, 0))

    def render(self):
        # pygame.init()
        # pygame.display.set_caption("BlueBoat Control") # set the title of the window
        # background = pygame.display.set_mode((self.screen_width, self.screen_height))
        #clock = pygame.time.Clock()
        # boat_img = pygame.transform.smoothscale( pygame.image.load("../img/bb.png").convert_alpha(), boat_img_size)
        # trailer_img = pygame.transform.smoothscale( pygame.image.load("../img/trailer.png").convert_alpha(), trailer_img_size)
        # pygame.init()
        # pygame.display.set_caption("BlueBoat Control") # set the title of the window
        
        # while not self.Done:
            #clock.tick(30)             # GUI refresh rate
                             
        for event in pygame.event.get():
        #if False:
            if event.type == pygame.QUIT:                    
                self.Done = True                                   
            if event.type == pygame.KEYDOWN:    # keyboard control
                if event.key == pygame.K_r:     # "r" key resets the simulator
                    self.control = [0,0]
                    # boat.reset()
                    self.reset()
                if event.key == pygame.K_p:     # holding "p" key freezes time
                    self.Pause = True
                if event.key == pygame.K_UP:
                    self.control[0] = self.control[0]+self.LINACCEL_INCR
                if event.key == pygame.K_DOWN:
                    self.control[0] = self.control[0]-self.LINACCEL_INCR
                if event.key == pygame.K_RIGHT:
                    self.control[1] = self.control[1]-self.ROTACCEL_INCR
                if event.key == pygame.K_LEFT:
                    self.control[1] = self.control[1]+self.ROTACCEL_INCR 
                if event.key == pygame.K_q:
                    self.Done = True
            if event.type == pygame.KEYUP:      # releasing "p" makes us live again
                if event.key == pygame.K_p:
                    self.Pause = False
            if event.type == pygame.JOYAXISMOTION:      # xbox joystick controller control
                if event.axis == 1:  # Left stick vertical axis = throttle
                    self.control[0] = self.JOY_MAX_LIN_ACCEL * -self.joystick.get_axis(1)
                if event.axis == 2:  # Right stick horizontal axis = steering 
                    self.control[1] = self.JOY_MAX_ROT_ACCEL * -self.joystick.get_axis(2)
                
        # if not self.Pause:
            # obs, r, done, trun, state = self.step(self.control)

        self.redraw(self.background, self.boat_img, self.trailer_img)
         
        # pygame.quit()

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
        # return a 4-tuple (obs, reward, done, info)

        self.u = u
        if dt is None:
            dt = 0.005
        t1 = self.solver.t + dt
        upper_t = 10.0
        while self.solver.successful and self.solver.t < t1:
        # while self.solver.successful and self.solver.t < t1 and self.solver.t < upper_t:
            self.solver.integrate(self.solver.t+ dt)
            
            # print("self.solver.t: ", self.solver.t)
            # print("t1: ", t1)
            # print("\n")
            
        self.x = np.array(self.solver.y , dtype=np.float32)
        self.t = self.solver.t
        
        observation = self.get_obs()
        reward = self.get_reward()
        reward = float(reward)
        # print("reward: ", reward)
        # print("action: ", self.u)
        done = False
        # if reward >= 1.0:
        # if not self.is_inside_map(self.x[0], self.x[1]):
        #     done = True
        truncated = False
        info = {"t": self.t, "action": self.u}
        
        # return ({"state": self.x}, reward, done, truncated, {"action": self.u})
        return observation, reward, done, truncated, info
