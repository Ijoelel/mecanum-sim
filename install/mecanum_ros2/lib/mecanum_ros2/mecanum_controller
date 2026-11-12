#!/usr/bin/env python3
# ROS 2 port of the mecanum controller / encoder-odometry node.
# subscribes: /cmd_vel, /joint_states, /imu
# publishes:  /odom, odom->base_footprint TF, wheel velocity commands
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


# --- small replacements for tf.transformations (ROS 1) ----------------------
def euler_from_quaternion(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class MecanumCtrl(Node):
    def __init__(self):
        super().__init__('mecanum_ctrl')

        # ---------------- physical parameters ----------------
        self.base_lx = 0.2
        self.base_ly = 0.288
        self.lx_ly = self.base_lx + self.base_ly
        self.wheel_radii = 0.04848
        self.encoder_radii = 0.019
        self.omni_radii = 0.04
        self.wheel_posn = [[0.2, 0.056, -0.236], [-0.2, 0.056, -0.34],
                           [-0.2, 0.056, 0.236], [0.2, 0.056, 0.34]]
        self.wheel_orient = [[1.57, 0.0, 0.0], [-1.57, 0.0, 0.0],
                             [-1.57, 0.0, 0.0], [1.57, 0.0, 0.0]]
        self.motor_joint_names = {'left_rim_one_joint': 0, 'right_rim_two_joint': 1,
                                  'left_rim_three_joint': 2, 'right_rim_four_joint': 3}
        self.encoder_joint_names = {'encoder_one_joint': 0, 'encoder_two_joint': 1,
                                    'encoder_three_joint': 2}
        self.wheel_frames = ['left_rim_one_link', 'right_rim_two_link',
                             'left_rim_three_link', 'right_rim_four_link']
        self.motor_joint_index = list(self.motor_joint_names.values())
        self.encoder_joint_index = list(self.encoder_joint_names.values())

        # ---------------- state ----------------
        self.curr_pos = np.array([0.0, 0.0, 0.0])  # x, y, yaw (odom frame)
        self.bf_x_pos = 0.0
        self.bf_y_pos = 0.0
        self.bf_yaw_z = 0.0
        self.motor_rotn = [[0.0] for _ in range(4)]
        self.encoder_rotn = np.zeros((3, 1))
        self.encoder_prev_rotn = np.zeros((3, 1))
        self.bot_acc = Vector3()
        self.bot_orientation = Quaternion()
        self.encoder_offset = [0.0] * 3
        self.motor_theta_offset = [0.0] * 4
        self.bool_motor_offset = True
        self.bf_index = -1
        self.x_error = 0.0
        self.y_error = 0.0
        self.w_error = 0.0
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.last_cmd_time = time.monotonic()
        self.ds_displacement = [0.0, 0.0, 0.0]
        self.first_update = True

        # ---------------- topics / frames ----------------
        self.odom_frame = 'odom'
        self.robot_base_frame = 'base_footprint'

        # ---------------- kinematics ----------------
        self.omni_w_to_v_mat = 1 / 3 * self.encoder_radii * np.array(
            [[0, -math.sqrt(3), math.sqrt(3)],
             [2, -1, -1],
             [1, 1, 1]])
        self.v_to_w_mat = 1 / self.wheel_radii * np.array(
            [[1, 1, self.lx_ly],
             [1, -1, self.lx_ly],
             [1, 1, -self.lx_ly],
             [1, -1, -self.lx_ly]])
        # match the rotation direction
        self.v_to_w_mat[1] *= -1
        self.v_to_w_mat[2] *= -1

        # ---------------- subscribers ----------------
        self.create_subscription(Twist, '/cmd_vel', self.callback_cmd, 10)
        self.create_subscription(Imu, '/imu', self.callback_imu, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.callback_joint_states, 10)

        # ---------------- publishers / tf ----------------
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        self.wheel_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/wheel_velocity_controller/commands',
            10
        )

        # 100 Hz main loop (replaces the blocking while/rate.sleep loop)
        self.update_timer = self.create_timer(0.01, self.update_bot)

        self.get_logger().info('Mecanum node started')

    # ---------------------------- callbacks ---------------------------
    def callback_cmd(self, msg: Twist):
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z
        self.last_cmd_time = time.monotonic()

    def callback_imu(self, msg: Imu):
        self.bot_orientation = msg.orientation
        self.bot_acc.x = msg.linear_acceleration.x
        self.bot_acc.y = msg.linear_acceleration.y
        self.bot_acc.z = msg.linear_acceleration.z
        o = self.bot_orientation
        _roll, _pitch, self.curr_pos[2] = euler_from_quaternion(o.x, o.y, o.z, o.w)

    def callback_joint_states(self, msg: JointState):
        position = msg.position
        if self.bool_motor_offset:
            names = list(msg.name)
            self.get_logger().info(str(names))
            for z, name in enumerate(names):
                if name in self.encoder_joint_names:
                    self.encoder_joint_names[name] = z
                if name in self.motor_joint_names:
                    self.motor_joint_names[name] = z
            self.motor_joint_index = list(self.motor_joint_names.values())
            self.encoder_joint_index = list(self.encoder_joint_names.values())
            for i in range(4):
                self.motor_theta_offset[i] = position[self.motor_joint_index[i]]
            for i in range(3):
                self.encoder_offset[i] = position[self.encoder_joint_index[i]]
            self.bool_motor_offset = False

        for i in range(4):
            self.motor_rotn[i][0] = position[self.motor_joint_index[i]] - self.motor_theta_offset[i]
            self.wheel_orient[i][1] = self.motor_rotn[i][0]  # for wheel tf
        for i in range(3):
            self.encoder_rotn[i][0] = position[self.encoder_joint_index[i]] - self.encoder_offset[i]

    # ----------------------------- tf ---------------------------------
    def publish_base_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.robot_base_frame
        t.transform.translation.x = float(self.curr_pos[0])
        t.transform.translation.y = float(self.curr_pos[1])
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(self.curr_pos[2]))
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def publish_wheel_tf(self, stamp):
        transforms = []
        for i in range(4):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.robot_base_frame
            t.child_frame_id = self.wheel_frames[i]
            t.transform.translation.x = float(self.wheel_posn[i][0])
            t.transform.translation.y = float(self.wheel_posn[i][1])
            t.transform.translation.z = float(self.wheel_posn[i][2])
            qx, qy, qz, qw = quaternion_from_euler(*self.wheel_orient[i])
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)
        self.tf_broadcaster.sendTransform(transforms)

    # --------------------------- main loop -----------------------------
    def update_bot(self):
        if time.monotonic() - self.last_cmd_time > 0.5:
            self.cmd_vel = [0.0, 0.0, 0.0]
            
        stamp = self.get_clock().now().to_msg()

        if self.first_update:
            self.encoder_prev_rotn = self.encoder_rotn.copy()
            self.first_update = False

        dth_encoder = self.encoder_rotn - self.encoder_prev_rotn
        self.encoder_prev_rotn = self.encoder_rotn.copy()

        ds_wrt_bot = np.matmul(self.omni_w_to_v_mat, dth_encoder)
        yaw = self.curr_pos[2]
        self.ds_displacement[0] = ds_wrt_bot[0][0] * math.cos(yaw) - ds_wrt_bot[1][0] * math.sin(yaw)
        self.ds_displacement[1] = ds_wrt_bot[0][0] * math.sin(yaw) + ds_wrt_bot[1][0] * math.cos(yaw)
        self.ds_displacement[2] = ds_wrt_bot[2][0] / self.omni_radii
        for j in range(2):
            self.curr_pos[j] += self.ds_displacement[j]

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.robot_base_frame
        odom.pose.pose.position.x = float(self.curr_pos[0])
        odom.pose.pose.position.y = float(self.curr_pos[1])
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(self.curr_pos[2]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        self.publish_base_tf(stamp)
        # self.publish_wheel_tf(stamp)

        vrr = np.array([[self.cmd_vel[0]], [self.cmd_vel[1]], [self.cmd_vel[2]]])
        prr = np.matmul(self.v_to_w_mat, vrr)
        self.wheel_cmd_pub.publish(
            Float64MultiArray(data=prr.flatten().tolist())
        )


def main(args=None):
    rclpy.init(args=args)
    node = MecanumCtrl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Keyboard interrupt received')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('Thank you for using')


if __name__ == '__main__':
    main()