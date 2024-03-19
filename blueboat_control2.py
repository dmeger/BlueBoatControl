import pygame
import numpy as np
import math
import sys
from envs import BlueBoat

class Controller:
    def __init__(self, controller_type):
        self.controller_type = controller_type
        if self.controller_type == "xbox":
            self.THROTTLE_AXIS = 1
            self.STEERING_AXIS = 2
            self.THROTTLE_MULTIPLIER = -1
            self.STEERING_MULTIPLIER = -1
            pygame.joystick.init() # initialize the joystick
            self.joystick = pygame.joystick.Joystick(0) # create a joystick object
            self.joystick.init() # initialize the joystick
            print("Using Xbox controller config.")
        elif self.controller_type == "ps4":
            self.THROTTLE_AXIS = 1
            self.STEERING_AXIS = 2
            self.THROTTLE_MULTIPLIER = -1
            self.STEERING_MULTIPLIER = -1
            pygame.joystick.init() # initialize the joystick
            self.joystick = pygame.joystick.Joystick(0) # create a joystick object
            self.joystick.init() # initialize the joystick
            print("Using PS4 controller config.")
        elif self.controller_type == "keyboard":
            print("Using keyboard config.")
        else:
            print("Invalid controller type. Please use 'keyboard', 'xbox' or 'ps4'.")
            sys.exit()

    def get_controller_throttle(self):
        throttle = self.joystick.get_axis(self.THROTTLE_AXIS) * self.THROTTLE_MULTIPLIER
        return throttle
    
    def get_controller_steering(self):
        steering = self.joystick.get_axis(self.STEERING_AXIS) * self.STEERING_MULTIPLIER
        return steering

# CONTROLLER
# Check for command-line argument specifying the controller type
controller_type = "keyboard"  # Default to Xbox controller
if len(sys.argv) > 1:
    controller_type = sys.argv[1].lower()

# Initialize the controller
controller = Controller(controller_type)

                           # (you can increase or decrease if you want to, just remind to keep even numbers)
Done = False                # if True,out of while loop, and close pygame
Pause = False               # when True, freeze the boat. This is 
                            # for debugging purposes

# Initialize the environment
x0 = [0,0,0,0,0,0]
env = BlueBoat(X0=x0)
env.reset()
env.render()

throttle = 0
steering = 0

# Main loop
while not Done:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:         # "r" key resets the simulator
                env.reset()
            if event.key == pygame.K_ESCAPE:    # "esc" key closes the simulator
                Done = True
            if event.key == pygame.K_UP:
                throttle = env.FORWARD_MAX_LIN_ACCEL
            if event.key == pygame.K_DOWN:
                throttle = -env.REVERSE_MAX_LIN_ACCEL
            if event.key == pygame.K_LEFT:
                steering = env.MAX_ROT_ACCEL
            if event.key == pygame.K_RIGHT:
                steering = -env.MAX_ROT_ACCEL
            if event.key == pygame.K_p:         # "p" key pauses the simulator
                Pause = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_p:         # releasing "p" key unpauses the simulator
                Pause = False
            if event.key == pygame.K_UP:
                throttle = 0
            if event.key == pygame.K_DOWN:
                throttle = 0
            if event.key == pygame.K_LEFT:
                steering = 0
            if event.key == pygame.K_RIGHT:
                steering = 0
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == controller.THROTTLE_AXIS:  # Left stick vertical axis = throttle
                throttle = controller.get_controller_throttle() * env.FORWARD_MAX_LIN_ACCEL
            if event.axis == controller.STEERING_AXIS: # Right stick horizontal axis = steering
                steering = controller.get_controller_steering() * env.MAX_ROT_ACCEL
    # If the environment is not paused, take action
    if not Pause:
        # Take action in the environment
        env.step([throttle, steering], None, False)
        # Render the environment
        env.render()

pygame.quit()