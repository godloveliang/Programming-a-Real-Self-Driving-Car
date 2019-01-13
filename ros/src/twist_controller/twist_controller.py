from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, wheel_base, wheel_radius, steer_ratio, min_speed, max_lat_accel, max_steer_angle, accel_limit, decel_limit, rate):      
        #parameters
        self.vehicle_mass = vehicle_mass
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.ts = 1.0/rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        
        #create throttle PID object
        kp = 0.3
        kd = 0.1
        ki = 0.1
        min = 0
        max = 0.3
        self.throttle_PID = PID(kp, ki, kd, min, max)
        
        #create velocity low pass object
        tau_vel = 0.1
        tau_err = 0.3
        self.vel_lpf = LowPassFilter(tau_vel, self.ts)
        self.error_lpf = LowPassFilter(tau_err, self.ts)
        
        #create yaw controller object
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
            

    def control(self, current_vel, dbw_enabled, linear_vel_cmd, angular_vel_cmd):
        
        if not dbw_enabled:
            self.throttle_PID.reset()
            return 0., 0., 0.
        
               
        filtered_velocity = self.vel_lpf.filt(current_vel)
        linear_vel_error = linear_vel_cmd - current_vel
        filtered_error = self.error_lpf.filt(linear_vel_error)
        
        steering = self.yaw_controller.get_steering(linear_vel_cmd, angular_vel_cmd, filtered_velocity)
        throttle = self.throttle_PID.step(filtered_error, self.ts)
        brake = 0.0
        
        if linear_vel_cmd == 0.0 and filtered_error < 0.1:
            throttle = 0.0
            brake = 700
        elif filtered_error < -0.05:
            throttle = 0.0
            decel = max(linear_vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius   
        elif throttle < 0.05:
            throttle = 0.0
            brake = 0.0
        
        return throttle, brake, steering
            
