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

    def get_controller_input(self):
        if self.controller_type == "xbox" or self.controller_type == "ps4":
            throttle = self.joystick.get_axis(self.THROTTLE_AXIS) * self.THROTTLE_MULTIPLIER
            steering = self.joystick.get_axis(self.STEERING_AXIS) * self.STEERING_MULTIPLIER
        elif self.controller_type == "keyboard":
            throttle = 0
            steering = 0
        return throttle, steering

    def get_controller_input(self):
        if self.controller_type == "xbox" or self.controller_type == "ps4":
            throttle = self.joystick.get_axis(self.THROTTLE_AXIS) * self.THROTTLE_MULTIPLIER
            steering = self.joystick.get_axis(self.STEERING_AXIS) * self.STEERING_MULTIPLIER
        elif self.controller_type == "keyboard":
            throttle = 0
            steering = 0
        return throttle, steering

    def computePIDControl(self, state, goalWP, linear_integral, angular_integral, previous_linear_error, previous_angular_error):
        # PID control
        Kp_lin = 10
        Ki_lin = 0.01
        Kd_lin = 30
        Kp_ang = 20
        Ki_ang = 0.01
        Kd_ang = 80
        # compute linear error as the euclidean distance between the boat and the goal
        linear_error = np.linalg.norm(goalWP - state[0:2])
        # print("linear_error: ", linear_error)
        # compute angular error as the difference between the angle of the line from the boat to the goal, and the current boat angle
        angular_error = - math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[2]
        # make sure the angular error is between -pi and pi
        angular_error = minangle(angular_error)
        # compute the integral of the linear and angular errors
        linear_integral += linear_error
        angular_integral += angular_error
        # anti-windup
        MAX_INTEGRAL_VALUE = 10
        if linear_integral > MAX_INTEGRAL_VALUE:
            linear_integral = MAX_INTEGRAL_VALUE
        if linear_integral < -MAX_INTEGRAL_VALUE:
            linear_integral = -MAX_INTEGRAL_VALUE
        if angular_integral > MAX_INTEGRAL_VALUE:
            angular_integral = MAX_INTEGRAL_VALUE
        if angular_integral < -MAX_INTEGRAL_VALUE:
            angular_integral = -MAX_INTEGRAL_VALUE
        # compute the derivative of the linear and angular errors
        linear_derivative = linear_error - previous_linear_error
        angular_derivative = angular_error - previous_angular_error
        # compute the linear and angular control inputs
        linear_control = (Kp_lin * linear_error + Ki_lin * linear_integral + Kd_lin * linear_derivative) * pow(1 - abs(angular_error / np.pi), 8) # the linear control is multiplied by 1 - abs(angular_error / np.pi) to reduce the linear control when the angular error is large
        angular_control = (Kp_ang * angular_error + Ki_ang * angular_integral + Kd_ang * angular_derivative)
        # store the current linear and angular errors for the next iteration
        previous_linear_error = linear_error
        previous_angular_error = angular_error
        # return the linear and angular control inputs
        return linear_control, angular_control, linear_integral, angular_integral, previous_linear_error, previous_angular_error

    def computePIDAngleControl(self, state, goal_yaw, angular_integral, previous_angular_error):
        # PID control
        Kp_ang = 20
        Ki_ang = 0.01
        Kd_ang = 80
        # compute angular error as the difference between the goal angle and the current boat angle
        angular_error = goal_yaw - state[2]
        # make sure the angular error is between -pi and pi
        angular_error = minangle(angular_error)
        # compute the integral of the angular error
        angular_integral += angular_error
        # anti-windup
        MAX_INTEGRAL_VALUE = 10
        if angular_integral > MAX_INTEGRAL_VALUE:
            angular_integral = MAX_INTEGRAL_VALUE
        if angular_integral < -MAX_INTEGRAL_VALUE:
            angular_integral = -MAX_INTEGRAL_VALUE
        # compute the derivative of the angular error
        angular_derivative = angular_error - previous_angular_error
        # compute the angular control input
        angular_control = (Kp_ang * angular_error + Ki_ang * angular_integral + Kd_ang * angular_derivative)
        # store the current angular error for the next iteration
        previous_angular_error = angular_error
        # return the angular control input
        return angular_control, angular_integral, previous_angular_error

    def waypointReached(self, state, goalWP, current_waypoint_index):
        # If the boat is within a certain distance of the current waypoint, move on to the next waypoint
        if self.pointReached(state, goalWP, 0.75):
            current_waypoint_index += 1
        return current_waypoint_index

    def pointReached(self, state, goalWP, threshold = 0.5):
        # If the boat is within a certain distance of the goal, return True
        if np.linalg.norm(goalWP - state[0:2]) < threshold:
            return True
        return False

def computeClampedControl(throttle, steering):
     # Update driving mode based on user input
    if abs(throttle) < THROTTLE_THRESHOLD:   # If throttle is not applied 
        current_mode = NEUTRAL
        clamped_throttle = 0
        clamped_steering = min(MAX_ROT_ACCEL, max(-MAX_ROT_ACCEL, steering)) # Clamp steering to valid range
    elif throttle > 0:          # If throttle is positive
        current_mode = FORWARD
        clamped_throttle = min(FORWARD_MAX_LIN_ACCEL, max(FORWARD_MIN_LIN_ACCEL, throttle)) # Clamp throttle to valid range
        clamped_throttle_ratio = (clamped_throttle / FORWARD_MAX_LIN_ACCEL)
        clamped_steering = min(MAX_ROT_ACCEL * clamped_throttle_ratio, max(-MAX_ROT_ACCEL * clamped_throttle_ratio, steering))
    elif throttle < 0:          # If throttle is negative
        current_mode = REVERSE
        clamped_throttle = - min(REVERSE_MAX_LIN_ACCEL, max(REVERSE_MIN_LIN_ACCEL, abs(throttle))) # Clamp throttle to valid range
        clamped_throttle_ratio = (- clamped_throttle / REVERSE_MAX_LIN_ACCEL)
        clamped_steering = min(MAX_ROT_ACCEL * clamped_throttle_ratio, max(-MAX_ROT_ACCEL * clamped_throttle_ratio, steering))
    else:
        current_mode = UNDEFINED
    return clamped_throttle, clamped_steering, current_mode


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
Auto_Control = False        # when True, the boat will follow the spiral path
                            # when False, the boat will be controlled by the user
Continuous_Control = False  # when True, the boat will be controlled continuously
Hide_Vel_Prof = True       # when True, the velocity profiles will not be displayed
Trailer_Following = False  # when True, the boat will follow the trailer
TFS = 0                    # Trailer Following State

class PathDrawer:
    def __init__(self, screen_center, screen_width, screen_height):
        self.screen_center = screen_center
        self.screen_width = screen_width
        self.screen_height = screen_height

    def draw_path(self, mode):
        match mode:
            case 0:
                return self.draw_spiral()
            case 1:
                return self.draw_spikes_coverage()
            case 2:
                return self.draw_square_coverage()
            case 3:
                return self.draw_slalom()
            case 4:
                return self.draw_large_slalom()
            case _:
                print("Invalid mode. Please use a number between 0 and 4.")
                return self.draw_spiral()

    def draw_spiral(self):
        spiral_points = []
        x = self.screen_center[0] 
        y = self.screen_center[1] 
        spiral_points.append((int(x), int(y)))
        spacing_factor = 8
        for t in np.arange(6 * np.pi, 15.5 * np.pi, 0.1):
            x = self.screen_center[0] + spacing_factor * t * np.cos(t)
            y = self.screen_center[1] + spacing_factor * t * np.sin(t)
            spiral_points.append((int(x), int(y)))
        return spiral_points

    def draw_spikes_coverage(self):
        coverage_points = []
        border_spacing = 120
        spacing = 120
        for x in range(0 + border_spacing, self.screen_width - border_spacing, spacing):
            for y in np.arange(0.0, 1.0, 0.1):
                coverage_points.append((x + y * spacing / 2, border_spacing + y * (self.screen_height - 2 * border_spacing)))
            for y in np.arange(0.0, 1.0, 0.1):
                coverage_points.append((x + (1 + y) * spacing / 2, self.screen_height - border_spacing - y * (self.screen_height - 2 * border_spacing)))
        return coverage_points

    def draw_square_coverage(self):
        coverage_points = []
        border_spacing = 100
        spacing = 200
        for x in range(0 + border_spacing, self.screen_width - border_spacing, spacing):
            for y in np.arange(0.0, 1.0, 0.1):
                coverage_points.append((x, border_spacing + y * (self.screen_height - 2 * border_spacing)))
            for y in np.arange(0.0, 1.0, 0.5):
                coverage_points.append((x + y * spacing / 2, self.screen_height - border_spacing))
            for y in np.arange(0.0, 1.0, 0.1):
                coverage_points.append((x + spacing / 2, self.screen_height - border_spacing - y * (self.screen_height - 2 * border_spacing)))
            for y in np.arange(0.0, 1.0, 0.5):
                coverage_points.append((x + (1 + y) * spacing / 2, border_spacing))
        return coverage_points

    def draw_slalom(self):
        slalom_points = []
        border_spacing = 60
        for x in range(0 + border_spacing, self.screen_width - border_spacing, 10):
            y = self.screen_center[1] + (self.screen_height - 2 * border_spacing) / 6 * np.sin((x - border_spacing) * 6 * np.pi / self.screen_width)
            slalom_points.append((x, y))
        return slalom_points

    def draw_large_slalom(self):
        slalom_points = []
        border_spacing = 60
        for x in range(0 + border_spacing, self.screen_width - border_spacing, 10):
            y = self.screen_center[1] + (self.screen_height - 2 * border_spacing) / 3 * np.sin((x - border_spacing) * 3 * np.pi / self.screen_width)
            slalom_points.append((x, y))
        return slalom_points

# A simple class to simulate BlueBoat physics using an ODE solver
class BlueBoat(object):


    def display_driving_mode(self, bg):
        # Display mode text in different colors based on the mode. Green background for forward, red background for reverse, blue background for neutral, black background for continuous, white background for undefined
        if current_mode == FORWARD:
            mode_text = font.render(current_mode, True, black, light_green)
        elif current_mode == REVERSE:
            mode_text = font.render(current_mode, True, black, light_red)
        elif current_mode == NEUTRAL:
            mode_text = font.render(current_mode, True, black, light_blue)
        elif current_mode == CONTINUOUS:
            mode_text = font.render(current_mode, True, white, black)
        elif current_mode == UNDEFINED:
            mode_text = font.render(current_mode, True, black, white)
        else:
            mode_text = font.render("Invalid mode", True, black, white)
        bg.blit(mode_text, driving_mode_position)

        
    def draw_velocity_profiles(self, bg):
        # draw 2 white rectanglse as the background for the velocity profiles
        pygame.draw.rect(bg, white, (lin_vel_prof_pos[0], lin_vel_prof_pos[1], vel_prof_width, vel_prof_height))
        pygame.draw.rect(bg, white, (ang_vel_prof_pos[0], ang_vel_prof_pos[1], vel_prof_width, vel_prof_height))
        # draw the x and y axes for the velocity profiles
        pygame.draw.line(bg, black, (lin_vel_prof_pos[0], lin_vel_prof_pos[1] + vel_prof_height), (lin_vel_prof_pos[0] + vel_prof_width, lin_vel_prof_pos[1] + vel_prof_height), 2)
        pygame.draw.line(bg, black, (lin_vel_prof_pos[0], lin_vel_prof_pos[1]), (lin_vel_prof_pos[0], lin_vel_prof_pos[1] + vel_prof_height), 2)
        pygame.draw.line(bg, black, (ang_vel_prof_pos[0], ang_vel_prof_pos[1] + vel_prof_height), (ang_vel_prof_pos[0] + vel_prof_width, ang_vel_prof_pos[1] + vel_prof_height), 2)
        pygame.draw.line(bg, black, (ang_vel_prof_pos[0], ang_vel_prof_pos[1]), (ang_vel_prof_pos[0], ang_vel_prof_pos[1] + vel_prof_height), 2)
        # draw the x and y axis labels for the velocity profiles
        linear_velocity_label = font.render("Linear Velocity (m/s)", True, black)
        bg.blit(linear_velocity_label, (lin_vel_prof_pos[0] + vel_prof_width / 2 - 80, lin_vel_prof_pos[1] + vel_prof_height - 30))
        angular_velocity_label = font.render("Angular Velocity (rad/s)", True, black)
        bg.blit(angular_velocity_label, (ang_vel_prof_pos[0] + vel_prof_width / 2 - 80, ang_vel_prof_pos[1] + vel_prof_height - 30))
        linear_velocity_profile = velocity_profiles[0]
        angular_velocity_profile = velocity_profiles[1]
        max_linear_velocity = math.ceil(max(linear_velocity_profile) + 0.1)
        min_linear_velocity = max(math.floor(min(linear_velocity_profile) - 0.1), 0)
        max_angular_velocity = math.ceil(max(angular_velocity_profile) + 0.1)
        min_angular_velocity = math.floor(min(angular_velocity_profile) - 0.1)
        lin_vel_range = max_linear_velocity - min_linear_velocity
        ang_vel_range = max_angular_velocity - min_angular_velocity
        lin_vel_tick_spacing = max(1, math.ceil(lin_vel_range / 5))
        ang_vel_tick_spacing = max(1, math.ceil(ang_vel_range / 5))
        # draw the ticks on the x and y axes for the velocity profiles
        for i in range(0, vel_prof_width, 50): 
            pygame.draw.line(bg, black, (ang_vel_prof_pos[0] + i, ang_vel_prof_pos[1] + vel_prof_height), (ang_vel_prof_pos[0] + i, ang_vel_prof_pos[1] + vel_prof_height + 5), 2)
        for i in range(min_linear_velocity, max_linear_velocity, lin_vel_tick_spacing): 
            pygame.draw.line(bg, black, (lin_vel_prof_pos[0] - 5, lin_vel_prof_pos[1] + vel_prof_height - (i - min_linear_velocity) / (max_linear_velocity - min_linear_velocity) * vel_prof_height), (lin_vel_prof_pos[0], lin_vel_prof_pos[1] + vel_prof_height - (i - min_linear_velocity) / (max_linear_velocity - min_linear_velocity) * vel_prof_height), 2)
        for i in range(min_angular_velocity, max_angular_velocity, ang_vel_tick_spacing): 
            pygame.draw.line(bg, black, (ang_vel_prof_pos[0] - 5, ang_vel_prof_pos[1] + vel_prof_height - (i - min_angular_velocity) / (max_angular_velocity - min_angular_velocity) * vel_prof_height), (ang_vel_prof_pos[0], ang_vel_prof_pos[1] + vel_prof_height - (i - min_angular_velocity) / (max_angular_velocity - min_angular_velocity) * vel_prof_height), 2)
        # display the tick labels
        for i in range(0, vel_prof_width, 50):
            label = font.render(str(i), True, black)
            bg.blit(label, (ang_vel_prof_pos[0] + i - 10, ang_vel_prof_pos[1] + vel_prof_height + 5))
        for i in range(min_linear_velocity, max_linear_velocity, lin_vel_tick_spacing):
            label = font.render(str(i), True, black)
            bg.blit(label, (lin_vel_prof_pos[0] - 30, lin_vel_prof_pos[1] + vel_prof_height - (i - min_linear_velocity) / (max_linear_velocity - min_linear_velocity) * vel_prof_height - 10))                                                                                    
        for i in range(min_angular_velocity, max_angular_velocity, ang_vel_tick_spacing):
            label = font.render(str(i), True, black)
            bg.blit(label, (ang_vel_prof_pos[0] - 30, ang_vel_prof_pos[1] + vel_prof_height - (i - min_angular_velocity) / (max_angular_velocity - min_angular_velocity) * vel_prof_height - 10))
        # draw the velocity profiles
        for i in range(len(linear_velocity_profile) - 1):
            pygame.draw.line(bg, light_green, (lin_vel_prof_pos[0] + i, lin_vel_prof_pos[1] + vel_prof_height - (linear_velocity_profile[i] - min_linear_velocity) / (max_linear_velocity - min_linear_velocity) * vel_prof_height), (lin_vel_prof_pos[0] + i + 1, lin_vel_prof_pos[1] + vel_prof_height - (linear_velocity_profile[i + 1] - min_linear_velocity) / (max_linear_velocity - min_linear_velocity) * vel_prof_height), 2)
            pygame.draw.line(bg, light_blue, (ang_vel_prof_pos[0] + i, ang_vel_prof_pos[1] + vel_prof_height - (angular_velocity_profile[i] - min_angular_velocity) / (max_angular_velocity - min_angular_velocity) * vel_prof_height), (ang_vel_prof_pos[0] + i + 1, ang_vel_prof_pos[1] + vel_prof_height - (angular_velocity_profile[i + 1] - min_angular_velocity) / (max_angular_velocity - min_angular_velocity) * vel_prof_height), 2)

    def add_to_velocity_profiles(self, boat_angle, x_velocity, y_velocity, angular_velocity):
        real_linear_velocity = np.sqrt(x_velocity ** 2 + y_velocity ** 2) # not necessarily in the same direction as the boat orientation
        velocity_profiles[0].append(real_linear_velocity)
        velocity_profiles[1].append(angular_velocity)
        if(len(velocity_profiles[0]) > vel_prof_width):
            velocity_profiles[0].pop(0)
            velocity_profiles[1].pop(0)
    
    def add_to_path_history(self, x, y):
        path_history.append((x, y))
        if(len(path_history) > 500):
            path_history.pop(0)

    def display_path_history(self, bg):
        for i in range(len(path_history) - 1):
            pygame.draw.line(bg, light_grey, self.to_screen(path_history[i][0], path_history[i][1]), self.to_screen(path_history[i + 1][0], path_history[i + 1][1]), 2)


# Clean up the screen and draw a fresh grid and the BlueBoat with its latest state coordinates
def redraw(timer): 
    background.fill(white)
    grid()
    boat.display_inside_trailer_info(background)
    if Auto_Control and current_waypoint_index < len(path_waypoints) - 1:
        for waypoint in path_waypoints[current_waypoint_index:]:
            pygame.draw.circle(background, black, waypoint, 3)
        pygame.draw.lines(background, red, False, path_waypoints[current_waypoint_index:], 2) # draw the path from the start of the spiral to current_waypoint_index
    boat.display_path_history(background)
    boat.draw(background)
    boat.draw_throttle_bar(background, throttle, clamped_throttle)
    boat.draw_steering_bar(background, steering, clamped_steering)
    boat.display_driving_mode(background)
    boat.display_inside_map_info(background)
    if not Hide_Vel_Prof:
        boat.draw_velocity_profiles(background)
     # Draw a solid blue circle in the center
    #pygame.draw.circle(background, (0, 0, 255), (250, 250), 75)
    timer_str = font.render("Time: " + str(timer), True, black)
    background.blit(timer_str, (screen_width - 100, 0))
    if timer % max_timer > 0.75 * max_timer:
        pos_change_warning = font.render("Trailer position will change in " + str(max_timer - timer % max_timer), True, black)
        background.blit(pos_change_warning, (screen_width - 500, 0))

    pygame.display.update()
        # Flip the display
    pygame.display.flip()

# Define driving modes
NEUTRAL = "NEUTRAL"
FORWARD = "FORWARD"
REVERSE = "REVERSE"
CONTINUOUS = "CONTINUOUS"
UNDEFINED = "UNDEFINED"

# Initialize variables
current_mode = NEUTRAL
auto_path_mode = 0
throttle = 0
steering = 0
clamped_throttle = 0
clamped_steering = 0
path_history = []
velocity_profiles = [[], []]

# Starting here is effectively the main function.
# It's a simple GUI drawing loop that calls to your code to compute the control, sets it to the 
# BlueBoat class and loops the GUI to show what happened.
boat = BlueBoat(x0) 
print(boat)
state = boat.get_state()
print(state)
path_drawer = PathDrawer(screen_center, screen_width, screen_height)
path_waypoints = path_drawer.draw_path(auto_path_mode)

FORWARD_MIN_LIN_ACCEL = 6.0
FORWARD_MAX_LIN_ACCEL = 16.0
REVERSE_MIN_LIN_ACCEL = 2.0
REVERSE_MAX_LIN_ACCEL = 8.0
MAX_ROT_ACCEL = 4.0
THROTTLE_THRESHOLD = 2.0

LINACCEL_INCR = 4.0
ROTACCEL_INCR = 2.0
control = [0,0]
current_waypoint_index = 0
linear_integral = 0
angular_integral = 0
previous_linear_error = 0
previous_angular_error = 0

timer = 0
max_timer = 2000
max_trailer_timer = 100
trailer_timer = 0
in_trailer = False

while not Done:
    #clock.tick(30)             # GUI refresh rate
   
    if timer % max_timer == 0 or trailer_timer > max_trailer_timer:
        trailer_pos, trailer_centre, trailer_yaw, trailer_approach_pt_1, trailer_approach_pt_0 = update_trailer_position()
        TFS = 0
        timer = 0
        trailer_timer = 0
        in_trailer = False
        
    for event in pygame.event.get():
    #if False:
        if event.type == pygame.QUIT:                    
            Done = True                                   
        if event.type == pygame.KEYDOWN:        # keyboard control
            if event.key == pygame.K_r:         # "r" key resets the simulator
                control = [0,0]
                path_history = []
                boat.reset()
            if event.key == pygame.K_p:         # holding "p" key freezes time
                Pause = True
            if event.key == pygame.K_a:         # "a" key toggles auto control
                Auto_Control = not Auto_Control
            if event.key == pygame.K_t:         # "t" key toggles auto trailer following
                Trailer_Following = not Trailer_Following
                TFS = 0
                in_trailer = False
            if event.key == pygame.K_v:         # "v" key toggles velocity profiles
                Hide_Vel_Prof = not Hide_Vel_Prof
            if event.key == pygame.K_c:         # "c" key toggles continuous control
                Continuous_Control = not Continuous_Control
            # set the auto path mode with numbers
            if event.key == pygame.K_1 or event.key == pygame.K_2 or event.key == pygame.K_3 or event.key == pygame.K_4 or event.key == pygame.K_5:
                auto_path_mode = int(event.unicode) - 1
                path_waypoints = path_drawer.draw_path(auto_path_mode)
                current_waypoint_index = 0
                path_history = []
                TFS = 0
                in_trailer = False
            if event.key == pygame.K_UP:
                control[0] = control[0]+LINACCEL_INCR
            if event.key == pygame.K_DOWN:
                control[0] = control[0]-LINACCEL_INCR
            if event.key == pygame.K_RIGHT:
                control[1] = control[1]-ROTACCEL_INCR
            if event.key == pygame.K_LEFT:
                control[1] = control[1]+ROTACCEL_INCR    
            if event.key == pygame.K_q:
                Done = True
        if event.type == pygame.KEYUP:          # releasing "p" makes us live again
            if event.key == pygame.K_p:
                Pause = False
        if event.type == pygame.JOYAXISMOTION:  # xbox joystick controller control
            if event.axis == THROTTLE_AXIS:     # Left stick vertical axis = throttle
                throttle = FORWARD_MAX_LIN_ACCEL * joystick.get_axis(THROTTLE_AXIS) * THROTTLE_MULTIPLIER
            if event.axis == STEERING_AXIS:     # Right stick horizontal axis = steering 
                steering = MAX_ROT_ACCEL * joystick.get_axis(STEERING_AXIS) * STEERING_MULTIPLIER

    if not Pause:
        #print(control)
        # use computePIDControl to compute the control input112
        if Auto_Control:
            if current_waypoint_index < len(path_waypoints) - 1:
                goal = boat.from_screen(path_waypoints[current_waypoint_index][0], path_waypoints[current_waypoint_index][1])
                throttle, steering, linear_integral, angular_integral, previous_linear_error, previous_angular_error = computePIDControl( state, goal, linear_integral, angular_integral, previous_linear_error, previous_angular_error )
                if Continuous_Control:
                    clamped_throttle, clamped_steering = throttle, steering
                    current_mode = CONTINUOUS
                else:
                    clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
                control = [clamped_throttle, clamped_steering]
                current_waypoint_index = waypointReached( state, goal, current_waypoint_index)
            else: # draw spiral again
                path_waypoints = path_drawer.draw_path(auto_path_mode)
                current_waypoint_index = 0
                path_history = []
        elif Trailer_Following:
            if TFS == 0: # to approach point 0
                goal = boat.from_screen(trailer_approach_pt_0[0], trailer_approach_pt_0[1])
                throttle, steering, linear_integral, angular_integral, previous_linear_error, previous_angular_error = computePIDControl( state, goal, linear_integral, angular_integral, previous_linear_error, previous_angular_error )
                if Continuous_Control:
                    clamped_throttle, clamped_steering = throttle, steering
                    current_mode = CONTINUOUS
                else:
                    clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
                control = [clamped_throttle, clamped_steering]
                if pointReached( state, goal, 0.3):
                    TFS = 1
            elif TFS == 1: # to approach point 1
                goal = boat.from_screen(trailer_approach_pt_1[0], trailer_approach_pt_1[1])
                throttle, steering, linear_integral, angular_integral, previous_linear_error, previous_angular_error = computePIDControl( state, goal, linear_integral, angular_integral, previous_linear_error, previous_angular_error )
                if Continuous_Control:
                    clamped_throttle, clamped_steering = throttle, steering
                    current_mode = CONTINUOUS
                else:
                    clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
                control = [clamped_throttle, clamped_steering]
                if pointReached( state, goal, 0.3):
                    TFS = 2
            elif TFS == 2: # to center
                goal = boat.from_screen(trailer_centre[0], trailer_centre[1])
                throttle, steering, linear_integral, angular_integral, previous_linear_error, previous_angular_error = computePIDControl( state, goal, linear_integral, angular_integral, previous_linear_error, previous_angular_error )
                if Continuous_Control:
                    clamped_throttle, clamped_steering = throttle, steering
                    current_mode = CONTINUOUS
                else:
                    clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
                control = [clamped_throttle, clamped_steering]
                if boat.is_in_trailer(state[0], state[1]):
                    TFS = 3
            elif TFS == 3: # to yaw
                goal_yaw = trailer_yaw
                throttle = 0
                steering, angular_integral, previous_angular_error = computePIDAngleControl( state, goal_yaw, angular_integral, previous_angular_error )
                if Continuous_Control:
                    clamped_steering, clamped_throttle = steering, throttle
                    current_mode = CONTINUOUS
                else:
                    clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
                control = [clamped_throttle, clamped_steering]
                if boat.is_in_trailer_with_yaw(state[0], state[1], state[2]):
                    if in_trailer == False:
                        in_trailer = True
                        trailer_timer = 0
                    else:
                        trailer_timer += 1
        elif throttle != 0 or steering != 0:
            clamped_throttle, clamped_steering, current_mode = computeClampedControl(throttle, steering)
            control = [clamped_throttle, clamped_steering]
        #control = computeControl( state )  # This is the call to the code you write
        state = boat.step(control)
        boat.add_to_path_history(state[0], state[1])
        boat.add_to_velocity_profiles(state[2], state[3], state[4], state[5])
        #print(state)

    redraw(timer)

    timer += 1
 
pygame.quit()