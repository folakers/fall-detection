import argparse
import os

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints, BODY_PARTS_PAF_IDS, BODY_PARTS_KPT_IDS
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def judgement_1(poses_queue, time_stamp, sightline):              #new mod
    is_fall = []
    poses_0 = poses_queue[0]
    poses_10 = poses_queue[10]
    nose_0 = []
    neck_0 = []
    rhip_0 = []
    lhip_0 = []
    lsho_0 = []
    rsho_0 = []
    rank_0 = []
    lank_0 = []
    for poses in poses_0:
        nose_0.append(poses.keypoints[0])
        neck_0.append(poses.keypoints[1])
        rhip_0.append(poses.keypoints[8])
        lhip_0.append(poses.keypoints[11])
        lsho_0.append(poses.keypoints[5])
        rsho_0.append(poses.keypoints[2])
        rank_0.append(poses.keypoints[10])
        lank_0.append(poses.keypoints[13])
    nose_10 = []
    neck_10 = []
    rhip_10 = []
    lhip_10 = []
    lsho_10 = []
    rsho_10 = []
    rank_10 = []
    lank_10 = []
    for poses in poses_10:
        nose_10.append(poses.keypoints[0])
        neck_10.append(poses.keypoints[1])
        rhip_10.append(poses.keypoints[8])
        lhip_10.append(poses.keypoints[11])
        lsho_10.append(poses.keypoints[5])
        rsho_10.append(poses.keypoints[2])
        rank_10.append(poses.keypoints[10])
        lank_10.append(poses.keypoints[13])

    max_down_by_people = []

    #new mod
    if sightline == 90:
        for i in range(min(len(poses_0), len(poses_10))):
            dist = (nose_10[i][1] - nose_0[i][1] + neck_10[i][1] - neck_0[i][1]) / 2
            # print(v)
            #dist_10 = (((rank_10[i][1]+lank_10[i][1])/2 - (rhip_10[i][1]+lhip_10[i][1])/2) ** 2 + ((rank_10[i][1]+lank_10[i][1])/2 - (rhip_10[i][0]+lhip_10[i][0])/2) ** 2) ** 0.5
            #dist_0 = (((rank_0[i][1]+lank_0[i][1])/2 - (rhip_0[i][1]+lhip_0[i][1])/2) ** 2 + ((rank_0[i][1]+lank_0[i][1])/2 - (rhip_0[i][0]+lhip_0[i][0])/2) ** 2) ** 0.5
            dist_head_0 = ((nose_0[i][1]-neck_0[i][1])**2 + (nose_0[i][0]-neck_0[i][0])**2)**0.5
            dist_body_0 = ((neck_0[i][0]-(rank_0[i][0]+lank_0[i][0])/2)**2 + (neck_0[i][1]-(rank_0[i][1]+lank_0[i][1])/2)**2)**0.5
            dist_head_10 = ((nose_10[i][1] - neck_10[i][1]) ** 2 + (nose_10[i][0] - neck_10[i][0]) ** 2) ** 0.5
            dist_body_10 = ((neck_10[i][0] - (rank_10[i][0] + lank_10[i][0]) / 2) ** 2 + (neck_10[i][1] - (rank_10[i][1]+ lank_10[i][1]) / 2) ** 2) ** 0.5
            if 6.8 * dist_head_0 < dist_body_0 and 6.8 * dist_head_10 < dist_body_10 and 25 * dist_head_0 > dist_body_0 and 25 * dist_head_10 > dist_body_10:
                is_fall.append(True)
            else:
                is_fall.append(False)
        return is_fall

    if sightline == 0:
        for i in range(10):
            for j in range(len(poses_queue[i])):
                if i == 0:
                    max_down_by_people.append(poses_queue[i][j].keypoints[1][1])
                elif j < len(max_down_by_people):
                    max_down_by_people[j] = max(max_down_by_people[j], poses_queue[i][j].keypoints[1][1])

        for i in range(min(len(poses_0), len(poses_10))):
            dist = (nose_10[i][1] - nose_0[i][1] + neck_10[i][1] - neck_0[i][1]) / 2
            # print(v)
            l_to_r_sho = 0.5 * (((lsho_10[i][1] - rsho_10[i][1]) ** 2 + (lsho_10[i][0] - rsho_10[i][0]) ** 2) ** 0.5
                                + ((lsho_0[i][1] - rsho_0[i][1]) ** 2 + (lsho_0[i][0] - rsho_0[i][0]) ** 2) ** 0.5)
            if abs(dist) > 1.4 * l_to_r_sho:
                if judgement_2(poses_queue, time_stamp, i, max_down_by_people, sightline):
                    is_fall.append(True)
                else:
                    is_fall.append(False)
            else:
                is_fall.append(False)
        return is_fall


def judgement_2(poses_queue, time_stamp, i, max_down_by_people, sightline):

    if sightline == 0:
        #最高值为nose
        #up_10 = []
        #down_10 = []
        #left_10 = []
        #right_10 = []
        #length = []
        #width = []

        #for j in range(len(poses_queue[10])):
        #找出其xy轴的最大最小值
        up_10 = poses_queue[10][i].keypoints[0][1]
        down_10 = max(poses_queue[10][i].keypoints[9][1], poses_queue[10][i].keypoints[12][1],
                      poses_queue[10][i].keypoints[10][1], poses_queue[10][i].keypoints[13][1])
        left_10 = min(poses_queue[10][i].keypoints[5][0], poses_queue[10][i].keypoints[0][0],
                      poses_queue[10][i].keypoints[2][0], poses_queue[10][i].keypoints[10][0],
                      poses_queue[10][i].keypoints[13][0])
        right_10 = max(poses_queue[10][i].keypoints[5][0], poses_queue[10][i].keypoints[0][0],
                       poses_queue[10][i].keypoints[2][0], poses_queue[10][i].keypoints[10][0],
                       poses_queue[10][i].keypoints[13][0])

        length = up_10-down_10
        width = right_10-left_10

        #if abs(length)/abs(width) > 0.63:
            #return False
        #else:
            #return True

        #if i >= len(poses_queue[20]) or i >= len(poses_queue[30]) or i >= len(poses_queue[40]):
            #return False
        if i >= len(poses_queue[20]) or i >= len(poses_queue[30]) or i >= len(poses_queue[40]):
            return False
        pose_i_20 = poses_queue[20][i]
        pose_i_30 = poses_queue[30][i]
        pose_i_40 = poses_queue[40][i]

        pose_i = [pose_i_20, pose_i_30, pose_i_40]

        if abs(length)/abs(width) < 0.63:
            if (pose_i[0].keypoints[1][1] > max_down_by_people[i] and pose_i[1].keypoints[1][1] > max_down_by_people[i] \
                    and pose_i[2].keypoints[1][1] > max_down_by_people[i]):
                return False
            else:
                return True
        return False

def run_demo(net, image_provider, height_size, cpu, track, smooth, sightline):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    is_fall = []
    show = []
    poses_queue = []
    count = 0
    time_stamp = []

    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        poses_queue.append(current_poses)
        time_stamp.append(time.time())

        count = count + 1

        #适用于多帧对比处理    nchange1
        if sightline == 0:
            if len(poses_queue) > 61:
                poses_queue.pop(0)
                time_stamp.pop(0)

            if len(poses_queue) == 61:
                # C1: 下降距离与肩比例
                # C2: 比最高值高一定时间
                # C3: 双肩，双髋保持
                is_fall = judgement_1(poses_queue, time_stamp, sightline)

            if len(poses_queue) == 61 and count % 50 == 0:
                show = is_fall

            for i in range(len(is_fall)):
                if i >= len(show):
                    show.insert(i, is_fall[i])
                elif (not show[i]) and is_fall[i]:
                    show[i] = True

            if len(poses_queue) == 61 and count % 70 == 0:
                for i in range(len(is_fall)):
                    if i < len(show):
                        show[i] = False

            if track:
                track_poses(previous_poses, current_poses, smooth=smooth)
                previous_poses = current_poses
            for pose in current_poses:
                pose.draw(img)

            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for i in range(len(current_poses)):
                # cv2.rectangle(img, (current_poses[i].bbox[0], current_poses[i].bbox[1]),
                #               (current_poses[i].bbox[0] + current_poses[i].bbox[2],
                #                current_poses[i].bbox[1] + current_poses[i].bbox[3]), (0, 255, 0))
                # .format(current_poses[i].id)
                if len(poses_queue) == 61 and i < len(show) and show[i]:
                    cv2.putText(img, 'WARNING!', (current_poses[i].bbox[0], current_poses[i].bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.imshow('Fall-detection Based upon OpenPose Algorithm', img)

        #单帧的图像处理          nchange2
        if sightline == 90:
            if len(poses_queue) > 11:
                poses_queue.pop(0)
                time_stamp.pop(0)

            if len(poses_queue) == 11:
                # C1: 下降距离与肩比例
                # C2: 比最高值高一定时间
                # C3: 双肩，双髋保持
                is_fall = judgement_1(poses_queue, time_stamp, sightline)

            if len(poses_queue) == 11 and count % 8 == 0:
                show = is_fall

            for i in range(len(is_fall)):
                if i >= len(show):
                    show.insert(i, is_fall[i])
                elif (not show[i]) and is_fall[i]:
                    show[i] = True

            if len(poses_queue) == 11 and count % 15 == 0:
                for i in range(len(is_fall)):
                    if i < len(show):
                        show[i] = False

            if track:
                track_poses(previous_poses, current_poses, smooth=smooth)
                previous_poses = current_poses
            for pose in current_poses:
                pose.draw(img)

            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for i in range(len(current_poses)):
                # cv2.rectangle(img, (current_poses[i].bbox[0], current_poses[i].bbox[1]),
                #               (current_poses[i].bbox[0] + current_poses[i].bbox[2],
                #                current_poses[i].bbox[1] + current_poses[i].bbox[3]), (0, 255, 0))
                # .format(current_poses[i].id)
                if len(poses_queue) == 11 and i < len(show) and show[i]:
                    cv2.putText(img, 'WARNING', (current_poses[i].bbox[0], current_poses[i].bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.imshow('Fall-detection Based upon OpenPose Algorithm', img)

        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Fall-detection Based upon OpenPose Algorithm''')
    parser.add_argument('--checkpoint-path', type=str, required=False, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    #new mod
    parser.add_argument('--sightline', type=int, default=0, help='optimize the algorithm')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    # new mod
    if not(args.sightline == 90 or args.sightline == 0):
        raise ValueError('use sightline from 0 or 90 to choose the right algorithm')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('./models/checkpoint.pth', map_location='cuda')  #change
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.sightline)
