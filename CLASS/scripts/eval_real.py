import time
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from PIL import Image as PIL_Image
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import roma
from collections import deque, defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import matplotlib.pyplot as plt
import json

from CLASS.util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d
from CLASS.util.common import conditional_resolver

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("if", conditional_resolver, replace=True)

try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print('ROS not available!')
try:
    from std_msgs.msg import Float32MultiArray, String
except ImportError:
    pass
try:
    from sensor_msgs.msg import Image
except ImportError:
    pass

def euler_to_rotation_matrix(euler_angles):
    """Convert euler angles to rotation matrix (assuming ZYX order)"""
    x, y, z = euler_angles
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def create_gripper_points(scale=0.01):
    """Create points for a -ㄷ shape rotated 90 degrees clockwise"""
    # Points for a rotated -ㄷ shape
    points = np.array([
        [0, 0, -scale],    # Left point of horizontal line
        [0, 0, 0],        # Center point where vertical line starts
        [0, scale, 0],    # Top point of vertical line
        [0, scale, scale], # Top right point
        [0, -scale, 0],   # Bottom point of vertical line
        [0, -scale, scale] # Bottom right point
    ])
    return points

def transform_gripper(points, position, euler_angles):
    """Transform gripper points by position and orientation"""
    R = euler_to_rotation_matrix(euler_angles)
    transformed_points = (R @ points.T).T + position
    return transformed_points

def plot_gripper_poses(poses, fig=None, ax=None):
    """
    Plot gripper poses in 3D
    poses: list of [x, y, z, rx, ry, rz] poses
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    base_points = create_gripper_points()
    
    for pose in poses:
        position = pose[:3]
        euler_angles = pose[3:]
        
        # Transform gripper points
        transformed_points = transform_gripper(base_points, position, euler_angles)
        
        # Plot initial horizontal line
        ax.plot3D([transformed_points[0][0], transformed_points[1][0]],
                  [transformed_points[0][1], transformed_points[1][1]],
                  [transformed_points[0][2], transformed_points[1][2]], 'r-')
        
        # Plot vertical line
        ax.plot3D([transformed_points[4][0], transformed_points[2][0]],
                  [transformed_points[4][1], transformed_points[2][1]],
                  [transformed_points[4][2], transformed_points[2][2]], 'r-')
        
        # Plot top horizontal line
        ax.plot3D([transformed_points[2][0], transformed_points[3][0]],
                  [transformed_points[2][1], transformed_points[3][1]],
                  [transformed_points[2][2], transformed_points[3][2]], 'r-')
        
        # Plot bottom horizontal line
        ax.plot3D([transformed_points[4][0], transformed_points[5][0]],
                  [transformed_points[4][1], transformed_points[5][1]],
                  [transformed_points[4][2], transformed_points[5][2]], 'r-')
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig, ax


class RealEnv:
    def __init__(self, control = "joints" ,mock_output=False, image_latency_frame = 0, device = 'cuda', control_width = False) -> None:
        
        self.ros_topic = {
            "front_camera": "/cam_front_image/color/image_raw",
            "robot_ee_state": "/teleoperation_server/current_state",
            "robot_joints_state": "/teleoperation_server/current_joints_state",
            "gripper_state_topic": "/teleoperation_server/gripper_state",
        }
        self.mock_output = mock_output
        self.image_latency_frame = image_latency_frame
        self.device = device
        if(self.mock_output):
            return
        self.control = control
        if(control == "joints"):
            self.goal_topic = "/teleoperation_server/goto/goal_state_joints"
        elif(control == "pose"):
            self.goal_topic = "/teleoperation_server/goto/goal_state"
        else:
            print(f"ERROR : {control}")
            exit(0)
        # self.frequency = frequency
        self.done = False
        self.current_obs_pil = None
        self.pause = False
        self.control_width = control_width
        self.node_state = ""
        self.cmd_topic = "/command_to_central_server"
        self.globla_cmd_topic = "/global_control_command"
        self.reset()
        self.init_subscriber()
        self.init_publisher()
        
    def reset(self):
        self.joints_states = deque([], maxlen = self.image_latency_frame+1)
        self.end_effector_states = deque([], maxlen = self.image_latency_frame+1)
        self.gripper_states = deque([], maxlen = self.image_latency_frame+1)

    def check_all_topics_received(self):
        if self.mock_output:
            return True
        if (
            self.if_received_ee_state
            and self.if_received_joints_state
            and self.if_received_front_image
        ):
            return True
        else:
            if not self.if_received_ee_state:
                print("ee state do not received")

            if not self.if_received_joints_state:
                print("raw state do not received")

            if not self.if_received_front_image:
                print("front image do not received")

            return False

    # def pulse(self, data):
    #     self.if_teleoperation_server_running = True

    # def init_frankapy_controller(self):
    #     self.PoseController = GotoPoseLive()
    #     self.init_deoxys_gripper_sockets()

    def init_publisher(self):
        self.pub_target_goal = rospy.Publisher(
            self.goal_topic, Float32MultiArray, queue_size=1
        )
        
        self.pub_cmd = rospy.Publisher(
            self.cmd_topic, String, queue_size=1
        )

    def init_subscriber(self):
        self.subscriber_ee_state = rospy.Subscriber(
            self.ros_topic["robot_ee_state"],
            Float32MultiArray,
            self.receive_ee_state,
            queue_size=1,
            buff_size=2**24,
        )
        self.subscriber_joints_state = rospy.Subscriber(
            self.ros_topic["robot_joints_state"],
            Float32MultiArray,
            self.receive_joints_state,
            queue_size=1,
            buff_size=2**24,
        )
        self.subscriber_front_image = rospy.Subscriber(
            self.ros_topic["front_camera"],
            Image,
            self.receive_front_image,
            queue_size=1,
            buff_size=2**24,
        )
        self.subscriber_gripper_state = rospy.Subscriber(
            self.ros_topic["gripper_state_topic"],
            Float32MultiArray,
            self.receive_gripper_state,
            queue_size=1,
            buff_size=2**24,
        )
        self.subscriber_global_control_command = rospy.Subscriber(self.globla_cmd_topic, String, self.receive_global_control_command, queue_size=1, buff_size=2**24)

        self.done = False
        self.if_received_ee_state = False
        self.if_received_joints_state = False
        self.if_received_front_image = False


    def send_command_to_central_server(self, command = "", command_dict = None):
        if(command_dict is None):
            if(command == "query_is_true"):
                command_dict = {
                    "node" : "central_server",
                    "from" : "dagger",
                    "command" : "switch_to_VR_control",
                    "pause" : True
                }
        else:
            command_dict = command_dict
            
        command_string = json.dumps(command_dict)
        msg = String(data = command_string)
        self.pub_cmd.publish(msg)

    def receive_global_control_command(self, data):
        print(data)
        command_dict = json.loads(data.data)
        command_node = command_dict["node"]
        if(command_node == "daggerEval" or command_node == "all"):
            if(command_dict["command"] == "done"):
                self.done = True
                print("receive_done")
            elif(command_dict["command"] == "pause"):
                self.node_state = "pause"
            elif(command_dict["command"] == "resume"):
                self.node_state = "resume"
            else:
                pass

    def receive_gripper_state(self, data):
        gripper_state = list(data.data)
        self.gripper_states.append(gripper_state)

    def receive_ee_state(self, data):
        self.if_received_ee_state = True
        end_effector_state = np.concatenate((data.data[0:3], data.data[7:10]))
        self.end_effector_states.append(end_effector_state)

    def receive_joints_state(self, data):
        self.if_received_joints_state = True
        self.joints_states.append(data.data)

    def process_image_topic_data(self, data):
        image_np = np.frombuffer(data.data, dtype=np.uint8)
        image_np = image_np.reshape(data.height, data.width, 3)
        return PIL_Image.fromarray(image_np)

    def receive_front_image(self, data):
        self.if_received_front_image = True
        self.front_image = self.process_image_topic_data(data)
        self.front_image.save("front_image_processed.jpg")
        
    @property
    def joints_state(self):
        return self.joints_states[0]

    @property
    def end_effector_state(self):
        return self.end_effector_states[0]

    @property
    def gripper_state(self):
        return self.gripper_states[0]

    def _get_obs(self):
        if self.mock_output:
            return {
                "front_image": PIL_Image.fromarray((np.random.rand(640, 480, 3) * 255).astype(np.uint8)),
                "joints_state": [0] * 7,
                "end_effector_state": [0] * 7,
                "gripper_state": [0] * 2,
            }
        else:
            end_effector_state = np.concatenate((self.end_effector_state[0:3], self.end_effector_state[3:6])).tolist() #position and euler
            # print("end_effector_state" , end_effector_state)
            gripper_state = list(self.gripper_states[0])

            obs = {
                "front_image": self.front_image,
                "robot_state": end_effector_state,                
                "joints_state": self.joints_state,
                "end_effector_state": end_effector_state,
                "gripper_state": gripper_state,
            }
        return obs
    
    def get_obs(self):
        obs = self._get_obs()
        obs['front_image'] = obs['front_image'].crop((80, 0, 560, 480))

        # Process gripper state (assuming it's a list or similar)
        obs['gripper_state'] = torch.tensor(list(obs['gripper_state']))

        end_effector_pos = torch.tensor(obs['end_effector_state'][:3])
        end_effector_euler = torch.tensor(obs['end_effector_state'][3:6]).float()
        roto_mat = roma.euler_to_rotmat('XYZ', end_effector_euler)

        from scipy.spatial.transform import Rotation as R
        rotation_eular = R.from_euler("xyz", np.array(obs['end_effector_state'][3:6]))

        rotation = R.from_matrix(np.array(roto_mat))
        quaternion_1 = rotation.as_quat()
        euler_angles = rotation.as_euler('xyz', degrees=False)
        # print("euler_angles", euler_angles)
        end_effector_6d = matrix_to_rotation_6d(roma.euler_to_rotmat('XYZ', end_effector_euler))
        obs['end_effector_state'] = torch.cat((end_effector_pos, end_effector_6d))

        # Resize all images to 224x224
        for key in obs.keys():
            if 'image' in key:
                obs[key] = obs[key].resize((256, 256))
                obs[key].save(f"{key}.jpg")
                obs[key] = T.functional.pil_to_tensor(obs[key]).to(self.device).unsqueeze(0)
            else:
                if(torch.is_tensor(obs[key])):
                    obs[key] = obs[key].to(self.device).unsqueeze(0)
                else:
                    obs[key] = torch.tensor(obs[key]).to(self.device).unsqueeze(0)
        return obs

    def step(self, action):
        if self.mock_output:
            obs = {
                "front_image": torch.zeros((1, 3, 224, 224)),
                "joints_state": torch.zeros((1, 7)),
                "end_effector_state": torch.zeros((1, 7)),
                "gripper_state": torch.zeros((1, 1)),
            }
            return obs, "", "", ""
        else:
            msg = Float32MultiArray(data=action)
            self.pub_target_goal.publish(msg)
            while (
                sum(abs(np.array(action[0:7]) - np.array(self.joints_state)))
                <= 0.001
            ):
                continue
            obs = {
                "front_image": self.front_image,
                "joints_state": self.joints_state,
                "end_effector_state": self.end_effector_state,
                "gripper_state": self.gripper_states,
            }
            
            return obs, "", "", ""

from hydra import compose, initialize
from hydra.utils import instantiate
with initialize(version_base=None, config_path="configs/"):
    gripper_delay_number = 12
    cfg = compose(config_name="grasp_cup_dp_pos")
    policy = instantiate(cfg.policy, _recursive_ = False) 
    try:
        test_mode = cfg.test_mode
    except:
        test_mode = False
    print(test_mode)
    try:
        interactive_mode = cfg.interactive_mode
    except:
        interactive_mode = False

    try:
        output_dir = cfg.running_output
    except:
        output_dir = "model_outputs"

    try:
        skip_msg_check = cfg.skip_msg_check
    except:
        skip_msg_check = False

    now = datetime.now()
    time_string = now.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    # Process and save image sequences as videos

    save_dir = os.path.join(output_dir , time_string)
    save_dir = os.path.join(os.path.abspath(os.getcwd()), save_dir)
    policy = hydra.utils.instantiate(cfg.policy, _recursive_ = False)

    print(cfg.load_file)
    print("---------------------")
    policy.load(cfg.load_file)
    policy.save_to_ema_model()
    
    if ROS_AVAILABLE:
        rospy.init_node("CLASS_eval")

    if "pos" in cfg.action_space:
        action_space = "joints" 
    elif "force" in cfg.action_space:
        action_space = "force" 
        import pytorch_kinematics as pk
        chain = pk.build_serial_chain_from_urdf(open("assets/panda_v3.urdf").read(), "panda_hand_tcp")
    else:
        raise ValueError(f'action space {cfg.action_space} not recognized.')
    env = RealEnv(control = action_space, mock_output=not ROS_AVAILABLE, image_latency_frame = cfg.image_latency_frame, device = cfg.device, control_width = True)

    while True and not skip_msg_check:
        if not env.check_all_topics_received():
            print("Some topics are not ready.")
            time.sleep(0.5)
        else:
            break

    obs = env.get_obs()
    print({k:v for k,v in obs.items() if 'image' not in k})
    print(policy.encode_obs({k:v[None] for k,v in obs.items() if k in policy.obs_keys}).max())
    obs_deques = deque([obs.copy()] * (policy.obs_horizon), maxlen=policy.obs_horizon)
    image_tensors = defaultdict(list)
    if test_mode:
        # Save the images
        for key, tensor in obs.items():
            if 'image' in key:  # Check if it's an image
                img_tensor = tensor[0]
                img = PIL_Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
                img.save(f"{key}.png")

        obs_seq = {key: torch.stack([obs_deques[j][key] for j in range(policy.obs_horizon)]).swapaxes(0, 1).float() for key in cfg.obs_keys}
        action = policy.get_action(obs_seq)[0][::2]
        if 'pose' in cfg.action_space:
            pos = action[:,:3]
            euler = roma.rotmat_to_euler('XYZ', rotation_6d_to_matrix(action[:,3:9]))
        else:
            ret = chain.forward_kinematics(action[:,:7].cpu(), end_only=False)
            tg = ret['panda_hand_tcp']
            m = tg.get_matrix()
            pos = m[:, :3, 3]
            euler = pk.matrix_to_euler_angles(m[:, :3, :3], 'XYZ')
            
        pose = torch.cat((pos, euler), dim = -1).cpu()
        fig, ax = plot_gripper_poses(pose.numpy().tolist())
        data = (pose[:,:3] - torch.tensor([0, 0, 0.01]))
        colors = plt.cm.RdBu(np.linspace(0, 1, len(data)))  # RdBu is red-to-blue colormap
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                            c=np.arange(len(data)),  # Color based on point index
                            cmap='winter',     # Red to Blue colormap
                            norm=plt.Normalize(0, len(data)-1))
        plt.colorbar(scatter, label='Time')
        ax.set_aspect('equal')
        plt.savefig('sim_trajectory.jpg')        
        print('trajectory generated in generated_trajectory.jpg')
        cfg.dataset.num_demo = 2
        test_dataset = instantiate(cfg.dataset)
        action = test_dataset.normalizers['action'].unnormalize(test_dataset[test_dataset.episode_starts[0]]['action'].cuda())[..., :-1].cpu()
        pos = action[:,:3]
        euler = roma.rotmat_to_euler('XYZ', rotation_6d_to_matrix(action[:,3:9]))
        pose = torch.cat((pos, euler), dim = -1)
        fig, ax = plot_gripper_poses(pose.numpy().tolist())
        data = (pose[:,:3] - torch.tensor([0, 0, 0.01]))
        colors = plt.cm.RdBu(np.linspace(0, 1, len(data)))  # RdBu is red-to-blue colormap
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                            c=np.arange(len(data)),  # Color based on point index
                            cmap='winter',     # Red to Blue colormap
                            norm=plt.Normalize(0, len(data)-1))
        plt.colorbar(scatter, label='Time')
        ax.set_aspect('equal')
        plt.savefig('dataset_trajectory.jpg')        
        print('trajectory generated in dataset_trajectory.jpg')
         
    else:
        gripper_count = gripper_delay_number
        for t in range(cfg.max_episode_steps * 2):
            if(env.done):
                break
            if(env.node_state == "pause"):
                print("PAUSE")
                while(1):
                    env.get_obs()
                    time.sleep(0.1)
                    if(env.node_state != "pause" or env.done):
                        break
            if(env.done):
                break
            obs_seq = {key: torch.stack([obs_deques[j][key] for j in range(policy.obs_horizon)]).swapaxes(0, 1).float() for key in cfg.obs_keys}
            actions = policy.get_action(obs_seq)[0][0:12]
            for action in actions:
                if "pose" in cfg.action_space:
                    pos, ori, gripper = action[:3], action[3:9], action[9:10]
                    ori  = rotation_6d_to_matrix(ori)
                    ori = roma.rotmat_to_euler('XYZ', ori)                
                else:
                    ret = chain.forward_kinematics(action[:,:7].cpu(), end_only=False)
                    tg = ret['panda_hand_tcp']
                    m = tg.get_matrix()
                    pos = m[:, :3, 3]
                    ori = pk.matrix_to_euler_angles(m[:, :3, :3], 'XYZ')
                
                action = torch.cat([pos, ori, gripper])   .cpu().numpy() 
                
                if(env.control_width):
                    action[-1] = action[-1]
                else:    
                    if action[-1] > 0.5:
                        gripper_count = gripper_delay_number
                    else:
                        gripper_count =  max(0, gripper_count - 1)
                    action[-1] =  1 if gripper_count else -1   
                time.sleep(.02)
                if(env.done):
                    break
                print("action", action)
                env.step(action)
                obs = env.get_obs()             
                obs_deques.append(obs)
                for key, tensor in obs.items():
                    if 'image' in key:
                        image_tensors[key].append(tensor[0])    

    try:
        os.makedirs(save_dir)
    except:
        pass

    command_dict = {
        "node" : "central_server",
        "from" : "CLASS",
        "command" : "prepare_data_collection",
        "directory" : save_dir,
        "time_string" : time_string,
        "file_id" : "_1"
    }

    env.send_command_to_central_server(command_dict = command_dict)

    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    for key, tensors in image_tensors.items():        
        images = [
            ((t.permute(1, 2, 0).cpu().numpy())).astype(np.uint8)
            for t in tensors
        ]

        if images:
            import imageio
            imageio.mimwrite(os.path.join(save_dir, f"{key}_{time_string}_video.mp4"), images, fps=30, codec='libx264')
            print(f"Video saved for {key}")

# if __name__ == "__main__":
#     main()