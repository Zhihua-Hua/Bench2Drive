import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from tqdm import trange, tqdm


def draw_spline_bev(meta, raw_img, canvas_size=(1024,1024),thickness=3,is_ego=False, type='navi', pred_frame=None, hue_start=120,hue_end=80):
    color_dict = {'navi': (51, 136, 255), 'pred': (255, 165, 0)}
    legend_dict = {'navi': (50, 50), 'pred': (50, 100)}
    rgb_color_tuple = color_dict[type]
    traj_type = 'navigation_points' if type=='navi' else 'planning_trajectory'

    line = np.array(meta[traj_type][:6])

    coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                            [0.0, -1.0,  0.0,  3.900000e-01], 
                            [0.0,  0.0, -1.0, 4.816000e+01], 
                            [0.0,  0.0,  0.0,  1.0]])
    # topdown_intrinsics = np.array([[1097.987543300894, 0.0, 512.0, 0], 
    #                             [0.0, 1097.987543300894, 512.0, 0], 
    #                             [0.0, 0.0, 1.0, 0], 
    #                             [0, 0, 0, 1.0]])
    topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], 
                                [0.0, 548.993771650447, 256.0, 0], 
                                [0.0, 0.0, 1.0, 0], 
                                [0, 0, 0, 1.0]])    # 分辨率为512x512
    coor2topdown = np.dot(topdown_intrinsics, coor2topdown)

    img = raw_img.copy()        
    pts_4d = np.stack([line[:,0],line[:,1],np.zeros(line.shape[0]),np.ones(line.shape[0])])
    pts_2d = (coor2topdown @ pts_4d).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
    if not mask.any():
        return img
    
    # draw raw points
    pts_2d = pts_2d[mask,0:2]

    try:
        tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=5) # s控制样条曲线插值误差
    except:
        return img
    unew = np.linspace(0, 1, 100)
    smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
    
    num_points = len(smoothed_pts)
    for i in range(num_points-1):
        hue = hue_start + (hue_end - hue_start) * (i / num_points)
        hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
        rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
        cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)  
    
    return img

def draw_traj_bev(meta, raw_img, canvas_size=(1024,1024),thickness=3,is_ego=False, type='navi', pred_frame=None):
    color_dict = {'navi': (51, 136, 255), 'pred': (255, 165, 0)}
    legend_dict = {'navi': (50, 50), 'pred': (50, 100)}
    rgb_color_tuple = color_dict[type]
    traj_type = 'navigation_points' if type=='navi' else 'planning_trajectory'

    line = np.array(meta[traj_type][:6])

    coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                            [0.0, -1.0,  0.0,  3.900000e-01], 
                            [0.0,  0.0, -1.0, 4.816000e+01], 
                            [0.0,  0.0,  0.0,  1.0]])
    # topdown_intrinsics = np.array([[1097.987543300894, 0.0, 512.0, 0], 
    #                             [0.0, 1097.987543300894, 512.0, 0], 
    #                             [0.0, 0.0, 1.0, 0], 
    #                             [0, 0, 0, 1.0]])
    topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], 
                                [0.0, 548.993771650447, 256.0, 0], 
                                [0.0, 0.0, 1.0, 0], 
                                [0, 0, 0, 1.0]])    # 分辨率为512x512
    coor2topdown = np.dot(topdown_intrinsics, coor2topdown)

    img = raw_img.copy()        
    pts_4d = np.stack([line[:,0],line[:,1],np.zeros(line.shape[0]),np.ones(line.shape[0])])
    pts_2d = (coor2topdown @ pts_4d).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
    if not mask.any():
        return img
    
    # draw raw points
    pts_2d = pts_2d[mask,0:2]
    for i in range(pts_2d.shape[0]):
        # rgb_color_tuple = (51, 136, 255) # 亮绿色
        if pts_2d[i,0]>0 and pts_2d[i,0]<canvas_size[1] and pts_2d[i,1]>0 and pts_2d[i,1]<canvas_size[0]:
            cv2.circle(img,(int(pts_2d[i,0]),int(pts_2d[i,1])),radius=4,color=rgb_color_tuple, thickness=thickness)   
        elif i==0:
            break
    
    legend_x, legend_y = legend_dict[type]
    cv2.circle(img, (legend_x, legend_y), radius=10, color=rgb_color_tuple, thickness=thickness)
    cv2.putText(img, type, (legend_x + 20, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)


    # draw line from smoothed points
    try:
        tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
    except:
        return img
    unew = np.linspace(0, 1, 100)
    smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
    num_points = len(smoothed_pts)
    for i in range(num_points-1):
        if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
            cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
        elif i==0:
            break
    return img

def create_video(scenario_path, video_path, fps=2, show_navi=False):
    # 检测什么时候car block
    # 标注txt, 标注轨迹, 合成前视图和俯视图

    # 获取图片列表
    images = [img for img in os.listdir(os.path.join(scenario_path, 'bev')) if img.endswith('.png') or img.endswith('.jpg')]
    images.sort()

    frame_img1 = cv2.imread(os.path.join(scenario_path, 'bev', images[0]))
    frame_img2 = cv2.imread(os.path.join(scenario_path, 'rgb_front', images[0]))
    # img2长宽各缩小2倍
    frame_img2 = cv2.resize(frame_img2, (int(frame_img2.shape[:2][1]/2), int(frame_img2.shape[:2][0]/2)))

    # 获取图片的原始尺寸
    height1, width1 = frame_img1.shape[:2]
    height2, width2 = frame_img2.shape[:2]
    target_height = max(height1, height2)
    target_width = width1 + width2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (target_width, target_height))

    for i in trange(1, len(images), leave=False):
        image_idx = images[i]
        meta = json.load(open(os.path.join(scenario_path,'meta', f'{i:04d}.json')))
        frame_img1 = cv2.imread(os.path.join(scenario_path, 'bev', image_idx))
        frame_img2 = cv2.imread(os.path.join(scenario_path, 'rgb_front', image_idx))

        # frame_img1
        # frame_img1 = draw_spline_bev(meta, frame_img1, is_ego=True, type='pred')
        frame_img1 = draw_traj_bev(meta, frame_img1, is_ego=True, type='pred')
        frame_img1 = draw_traj_bev(meta, frame_img1, is_ego=True, type='navi')
        # frame_img1 = draw_spline_bev(meta, frame_img1, is_ego=True, type='navi')

        # img2长宽各缩小2倍, 添加文字
        frame_img2 = cv2.resize(frame_img2, (int(frame_img2.shape[:2][1]/2), int(frame_img2.shape[:2][0]/2)))
        cv2.putText(frame_img2, f'frame {i:04d}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_img2, f'throttle: {meta["throttle"]:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_img2, f'brake: {meta["brake"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        black_background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        black_background[:height1, :width1] = frame_img1
        black_background[:height2, width1:width1+width2] = frame_img2
        video.write(black_background)
    video.release()


if __name__ == '__main__':
    results_path = '/data/huazh/Bench2Drive/Bench2Drive/eval_output/padding-road0.5-dropout0.4_qwen/checkpoint-22014/merged.json'
    results = json.load(open(results_path))

    success_path = '/data/huazh/Bench2Drive/220results/success'
    fail_path = '/data/huazh/Bench2Drive/220results/fail'
    os.makedirs(success_path, exist_ok=True)
    os.makedirs(fail_path, exist_ok=True)
    # 清空这两个文件夹
    for file in os.listdir(success_path):
        os.remove(os.path.join(success_path, file))
    for file in os.listdir(fail_path):
        os.remove(os.path.join(fail_path, file))
    root_path = '/data/huazh/Bench2Drive/Bench2Drive/eval_output/padding-road0.5-dropout0.4_qwen/checkpoint-22014'

    # 统计成功的场景；失败的场景
    # 分别生成对应的视频文件
    success_list = []
    fail_list = []
    for record in tqdm(results['_checkpoint']['records'], desc="Processing records"):
        if record['scores']['score_composed'] == 100:
            success_list.append(record['save_name'])

            # 获取场景路径
            result_path = None
            for i in range(7):
                prefix = f'bench2drive220_{i}_qwen_traj_'
                local_save_name = prefix + record['save_name']
                local_result_path = os.path.join(root_path, local_save_name)
                if os.path.exists(local_result_path):
                    result_path = local_result_path
                    break
            # 获取video name
            video_name = record['save_name'].split('_')[1] + "_" + record['scenario_name'].split('_')[0] +'_'+ record['weather_id']+ '.mp4'
            video_path = os.path.join(success_path, video_name)
            # 生成视频
            create_video(result_path, video_path)

        else:
            fail_list.append(record['save_name'])

            # 获取场景路径
            result_path = None
            for i in range(7):
                prefix = f'bench2drive220_{i}_qwen_traj_'
                local_save_name = prefix + record['save_name']
                local_result_path = os.path.join(root_path, local_save_name)
                if os.path.exists(local_result_path):
                    result_path = local_result_path
                    break
            # 获取video name
            video_name = record['save_name'].split('_')[1] + "_" + record['scenario_name'].split('_')[0] +'_'+ record['weather_id']+ '.mp4'
            video_path = os.path.join(fail_path, video_name)
            # 生成视频
            create_video(result_path, video_path)


    print('失败场景数：', len(fail_list))
    print('成功场景数：', len(success_list))

    # 生成对应场景的视频保存在两个文件夹下

    pass
