import gymnasium
import carla
import queue
import time
import os
from random import choice
import numpy as np
from gymnasium import spaces

class CarlaLaneTrackingEnv(gymnasium.Env):
    def __init__(self):
        super(CarlaLaneTrackingEnv, self).__init__()
        
        self.dev = 0
        self.timestep = 0
        self.spawn_idx = 0

        # Connect to Carla server
        self.client = carla.Client('192.168.0.10', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town07')
        self.map = self.world.get_map()
        
        self.spectator = self.world.get_spectator()
        
        # Setup blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle')[0]
        
        # ImageBuffer
        self.image_buffer = queue.Queue(maxsize=1)

        self.vehicle = None
        self.sensor = None
        self.col_sensor = None

        # Get spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()
        #self.spawn_idx_lst = [35, 38, 25, 54]
        self.spawn_idx_lst = [38]
        #self.spawn_idx_lst = [i for i in range(len(self.world.get_map().get_spawn_points()) - 1)]
        self.spawn_point = self.spawn_points[self.spawn_idx_lst[0]]

        # Setup action spaces and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(160, 320, 3), dtype=np.uint8)

        # Setup states
        self.done = False
        self.collide = False
        
    def _setup_sensors(self):
        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.col_sensor.listen(self._on_collision)

        # Segmentation Camera
        cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute('image_size_x', '320')
        cam_bp.set_attribute('image_size_y', '160')
        
        cam_transform = carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10))
        self.sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.sensor.listen(self.image_buffer.put)
        
    def _on_collision(self, event):
        self.collide = True
        
    def render(self, render_mode='human'):
        pass
        
    def spect(self):
        transform = self.vehicle.get_transform()

        # For Training(Top View)
        '''
        spectator_transform = carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
        '''
        
        # For Testing1 (1st View)
        offset = carla.Location(x=0.5, z=1.7)
        new_transform = carla.Transform(transform.location + transform.get_forward_vector() * offset.x + carla.Location(z=offset.z), transform.rotation)
        
        # For Testing2 (3rd View)
        '''
        offset = carla.Location(x=-3.5, z=7)
        new_transform = carla.Transform(transform.location + transform.get_forward_vector() * offset.x + carla.Location(z=offset.z), carla.Rotation(pitch=-40, yaw=transform.rotation.yaw))
        '''

        self.spectator.set_transform(new_transform)

    def reset(self, seed=0):
        self.collide = False
        self.done = False
        
        #self.spawn_point = self.spawn_points[choice(self.spawn_idx_lst)]
        self.spawn_point = self.spawn_points[self.spawn_idx_lst[self.spawn_idx % len(self.spawn_idx_lst)]]

        # Delete all instance
        self.close()

        # Respawn vehicle and sensors
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        self._setup_sensors()

        self.spect()
        
        return self._process_image(), {}

    def step(self, action):
        self.timestep += 1
        if self.timestep % (4096 * 4) == 0:
            self.spawn_idx += 1
        
        # For training(do not fix throttle)
        #self.vehicle.apply_control(carla.VehicleControl(throttle=float(action[0]), steer=float(action[1])))

        # For testing(fixed throttle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(0.5), steer=float(action[1])))

        # Compute reward and state
        reward = self._calculate_reward()
        self.done = self._check_done()
        self.spect()
        
        return self._process_image(), reward, self.done, False, {}

    def _process_image(self):
        while self.image_buffer.qsize() == 0:
            time.sleep(0.0001)
        
        image = self.image_buffer.get()
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        segmentation_mask = image_data[:, :, 2]

        # Generate blank black image
        result_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)

        # Coloring lane and road(Lane: Red, Road: Blue)
        result_image[segmentation_mask == 24] = [255, 0, 0]
        result_image[segmentation_mask == 1] = [0, 255, 0]
        
        return result_image
        
    def _calculate_reward(self):
        # Compute reward based deviation
        self.dev = self._calculate_lane_deviation()
        if self.dev > 1.0:
            return -1
        
        return max(1.0 - self.dev, 0)
        
    def _calculate_lane_deviation(self):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        waypoint = self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_center_location = waypoint.transform.location

        deviation = np.sqrt((vehicle_location.x - lane_center_location.x) ** 2 + (vehicle_location.y - lane_center_location.y) ** 2)
                            
        return deviation

    def _check_done(self):
        done = self.collide or self.dev > 1.2 or self.timestep >= 4096 * 4
        return done

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.sensor is not None:
            self.sensor.destroy()
        if self.col_sensor is not None:
            self.col_sensor.destroy()
