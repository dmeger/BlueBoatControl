import pygame
import numpy as np
import math
import sys
from utils import minangle
# import BlueBoat from envs
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
        if self.controller_type == "xbox" or self.controller_type == "ps4":
            throttle = self.joystick.get_axis(self.THROTTLE_AXIS) * self.THROTTLE_MULTIPLIER
        elif self.controller_type == "keyboard":
            throttle = 0
        return throttle
    
    def get_controller_steering(self):
        if self.controller_type == "xbox" or self.controller_type == "ps4":
            steering = self.joystick.get_axis(self.STEERING_AXIS) * self.STEERING_MULTIPLIER
        elif self.controller_type == "keyboard":
            steering = 0
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
# Main loop
while not Done:
    # Get controller input
    throttle = 0
    steering = 0

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                Done = True
            if event.key == pygame.K_p:
                Pause = not Pause
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_p:
                Pause = not Pause
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 1:  # Left stick vertical axis = throttle
                throttle = controller.get_controller_throttle()
            if event.axis == 2:
                steering = controller.get_controller_steering()
    # If the environment is not paused, take action
    if not Pause:
        # Take action in the environment
        action = np.array([throttle, steering])
        env.step(action)
        # Render the environment
        env.render()

    # If the environment is paused, wait for the user to unpause
    while Pause:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_p:
                    Pause = not Pause
pygame.quit()