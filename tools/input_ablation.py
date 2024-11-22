
import os
import re
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from scipy.interpolate import splprep, splev
from llava.model.builder import load_pretrained_model
from llava.eval.evaluation import EvalArguments, update_args
from llava.mm_utils import tokenizer_image_token, tokenizer_token, process_images
from llava.train.dataset import preprocess, DataCollatorForSupervisedDataset
from llava.constants import (
    TEXT_INPUT_CLIP_IMG,
    TEXT_INPUT_PERCEPTION,
    TEXT_INPUT_OBJ,
    TEXT_ANSWER_EGO_TRAJ,
    TEXT_INPUT_HEAD,
    TEXT_TASK,
    TEXT_PROMPT,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PERCEPTION_TOKEN,
    DEFAULT_OBJ_TOKEN,
    DEFAULT_EGO_TRAJ_TOKEN,
    TEXT_INPUT_MAP,
    DEFAULT_MAP_TOKEN,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# import debugpy
# print('waiting for debugger...')
# debugpy.listen(1991)
# debugpy.wait_for_client()

zero_state = [
        [
            -0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        [
            -0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        [
            -0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        [
            -0.0,
            0.0,
            0.0,
            0.0,
            0.039,
            -0.001,
            0.0,
            -0.0
        ]
    ]
state_71 = [
        [
            -0.275,
            -1.038,
            0.402,
            0.0,
            -17.594,
            -0.06,
            -0.012,
            -0.06
        ],
        [
            -0.338,
            -0.315,
            0.385,
            0.0,
            6.539,
            -0.01,
            0.001,
            0.012
        ],
        [
            -0.509,
            -1.11,
            0.904,
            0.0,
            3.886,
            0.305,
            -0.007,
            0.03
        ],
        [
            0.0,
            0.0,
            -0.001,
            0.0,
            0.089,
            -0.021,
            0.001,
            0.007
        ]
    ]
state_70 = [
        [
            -1.115,
            -0.907,
            0.61,
            0.0,
            3.461,
            0.249,
            0.003,
            0.008
        ],
        [
            0.235,
            0.072,
            0.402,
            0.0,
            -17.594,
            -0.06,
            -0.012,
            -0.06
        ],
        [
            0.173,
            0.795,
            0.385,
            0.0,
            6.539,
            -0.01,
            0.001,
            0.012
        ],
        [
            0.0,
            -0.0,
            0.904,
            0.0,
            3.886,
            0.305,
            -0.007,
            0.03
        ]
    ]

class AblationAgent():
    """
    初始化一个model
    根据scenario name和idx获取对应的输入数据
    调用model进行推理
    推理结果可视化
    """
    def __init__(self, model_path, model_name, torch_dtype, attn_implementation, scenario_path):
        self.scenario_path = scenario_path
        # self.device = "cuda"
        # self.tokenizer, self.model, _, _ = load_pretrained_model(model_path, model_name, torch_dtype, attn_implementation)
        # self.model.eval()
        # self.model.cuda()
        # cfg = EvalArguments()
        # self.cfg = update_args(cfg, self.model)
        # self.collate_fn = DataCollatorForSupervisedDataset(self.tokenizer)

    def draw_traj_bev(self, meta, raw_img, canvas_size=(1024,1024),thickness=3,is_ego=False, type='navi', pred_frame=None):
        color_dict = {'navi': (51, 136, 255), 'pred': (255, 165, 0)}
        legend_dict = {'navi': (50, 50), 'pred': (50, 100)}
        rgb_color_tuple = color_dict[type]
        traj_type = 'navigation_points' if type=='navi' else 'planning_trajectory'
        if is_ego:
            line = np.concatenate([np.zeros((1,2)),meta[traj_type]],axis=0)
            # line = np.array(meta['planning_trajectory'])
        else:
            line = meta[traj_type]

        # # 
        # if pred_frame is not None:
        #     traj = np.zeros((9, 2))
        #     traj = pred_frame*0.5 + line*0.5
        # else:
        #     traj = line
            
        # pred_frame = traj

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
    
    # def draw_traj_bev(self, traj, raw_img, meta, canvas_size=(1024,1024),thickness=3,is_ego=False, type='navi'):
    #     color_dict = {'navi': (51, 136, 255), 'pred': (255, 165, 0)}
    #     legend_dict = {'navi': (50, 50), 'pred': (50, 100)}
    #     rgb_color_tuple = color_dict[type]
    #     if is_ego:
    #         line = np.concatenate([np.zeros((1,2)),traj],axis=0)
    #         line = np.array(traj)
    #     else:
    #         line = traj

    #     coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
    #                             [0.0, -1.0,  0.0,  3.900000e-01], 
    #                             [0.0,  0.0, -1.0, 4.816000e+01], 
    #                             [0.0,  0.0,  0.0,  1.0]])
    #     topdown_intrinsics = np.array([[1097.987543300894, 0.0, 512.0, 0], 
    #                                 [0.0, 1097.987543300894, 512.0, 0], 
    #                                 [0.0, 0.0, 1.0, 0], 
    #                                 [0, 0, 0, 1.0]])
    #     coor2topdown = np.dot(topdown_intrinsics, coor2topdown)

    #     img = raw_img.copy()        
    #     pts_4d = np.stack([line[:,0],line[:,1],np.zeros(line.shape[0]),np.ones(line.shape[0])])
    #     pts_2d = (coor2topdown @ pts_4d).T
    #     pts_2d[:, 0] /= pts_2d[:, 2]
    #     pts_2d[:, 1] /= pts_2d[:, 2]
    #     mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
    #     if not mask.any():
    #         return img
        
    #     # draw raw points
    #     pts_2d = pts_2d[mask,0:2]
    #     for i in range(pts_2d.shape[0]):
    #         # rgb_color_tuple = (51, 136, 255) # 亮绿色
    #         if pts_2d[i,0]>0 and pts_2d[i,0]<canvas_size[1] and pts_2d[i,1]>0 and pts_2d[i,1]<canvas_size[0]:
    #             cv2.circle(img,(int(pts_2d[i,0]),int(pts_2d[i,1])),radius=4,color=rgb_color_tuple, thickness=thickness)   
    #         elif i==0:
    #             break
        
    #     legend_x, legend_y = legend_dict[type]
    #     throttle = f'throttle:{meta["throttle"]:.2f}'
    #     brake = f'brake:{meta["brake"]:.2f}'
    #     cv2.circle(img, (legend_x, legend_y), radius=10, color=rgb_color_tuple, thickness=thickness)
    #     cv2.putText(img, type, (legend_x + 20, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    #     # cv2.putText(img, throttle, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #     # cv2.putText(img, brake, (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    #     # draw line from smoothed points
    #     try:
    #         tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
    #     except:
    #         return img
    #     unew = np.linspace(0, 1, 100)
    #     smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

    #     num_points = len(smoothed_pts)
    #     for i in range(num_points-1):
    #         if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
    #             cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
    #         elif i==0:
    #             break
    #     return img

    def create_video(self, fps=2, show_navi=False):
        # 检测什么时候car block
        # 标注txt, 标注轨迹, 合成前视图和俯视图

        # 获取图片列表
        images = [img for img in os.listdir(os.path.join(self.scenario_path, 'bev')) if img.endswith('.png') or img.endswith('.jpg')]
        images.sort()

        frame_img1 = cv2.imread(os.path.join(self.scenario_path, 'bev', images[0]))
        frame_img2 = cv2.imread(os.path.join(self.scenario_path, 'rgb_front', images[0]))
        # img2长宽各缩小2倍
        frame_img2 = cv2.resize(frame_img2, (int(frame_img2.shape[:2][1]/2), int(frame_img2.shape[:2][0]/2)))

        # 获取图片的原始尺寸
        height1, width1 = frame_img1.shape[:2]
        height2, width2 = frame_img2.shape[:2]
        target_height = max(height1, height2)
        target_width = width1 + width2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        pattern = r"RouteScenario_\d+"  # 匹配 "RouteScenario_" 后跟一个或多个数字

        # 使用 re.search() 查找匹配的部分
        match1 = re.search(pattern, self.scenario_path)
        video = cv2.VideoWriter(f'{match1.group()}.mp4', fourcc, fps, (target_width, target_height))


        for i in trange(1, len(images)):
            image_idx = images[i]
            meta = json.load(open(os.path.join(self.scenario_path,'meta', f'{i:04d}.json')))
            frame_img1 = cv2.imread(os.path.join(self.scenario_path, 'bev', image_idx))
            frame_img2 = cv2.imread(os.path.join(self.scenario_path, 'rgb_front', image_idx))

            # frame_img1
            frame_img1 = self.draw_traj_bev(meta, frame_img1, is_ego=True, type='pred')
            # frame_img1 = self.draw_traj_bev(meta, frame_img1, is_ego=True, type='navi')

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

    def ablation_ego_state(self, data_dict):
        # ego state: x, y, vx, vy, ax, ay, rotation_ratex, ..._y
        # ego state 添加运动项

        # 4，8的tensor，将[:, 2:]全部取绝对值
        # data_dict['ego_state'][:, 2:] = torch.abs(data_dict['ego_state'][:, 2:])
        data_dict['ego_state'] = torch.tensor(zero_state)
        return data_dict

    def ablation_navigation(self, data_dict):
        # 更改navigation数据
        gen_step = 15
        x = torch.randn(gen_step)*0.5  # 10个均值为0，方差为1的随机数
        # 生成10个均值为0，方差为0.5的随机数
        i = torch.arange(gen_step, dtype=torch.float32)  # 生成0到9的索引
        y = 1 + 5 * i + torch.randn(gen_step)*0.5  # y = 5 * i + 随机噪声

        # 将x和y合并成一个10x2的张量
        gen_navi = torch.stack((x, y), dim=1)

        # 将trajectory扩充为16，2的张量，使用最后一个数值填充
        last_value = gen_navi[-1, :]
        expaneded_navi = torch.cat([gen_navi, last_value.unsqueeze(0).repeat(16-gen_step, 1)], dim=0)
        data_dict['navi_data'] = expaneded_navi
        print('navi_data: \n', data_dict['navi_data'])
        
        return data_dict
    

    def inference(self, idx):
        print(f"Inference on scenario:\n {self.scenario_path} \nwith idx {idx}")
        origin_dict = self.get_inputdict(idx)
        navi_dict = self.ablation_navigation(origin_dict)
        state_dict = self.ablation_ego_state(navi_dict)

        data_dict = state_dict

        input_data_batch = self.collate_fn([data_dict])

        for key, data in input_data_batch.items():
            if key != 'img_metas' and data is not None:
                if torch.is_tensor(data):
                    if data.dtype in [torch.float32, torch.float64, torch.float16]:
                        input_data_batch[key] = input_data_batch[key].to(device=self.model.device, dtype=self.model.dtype)
                    else:
                        input_data_batch[key] = input_data_batch[key].to(device=self.model.device)
                elif torch.is_tensor(data[0]):
                    if data[0].dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                        input_data_batch[key][0] = input_data_batch[key][0].to(device=self.model.device, dtype=self.model.dtype)
                    else:
                        input_data_batch[key][0] = input_data_batch[key][0].to(device=self.model.device)
        with torch.no_grad():
            pred_traj = self.model(**input_data_batch)
            pred_traj = pred_traj[0].cpu().float().numpy()

        return pred_traj

    def get_inputdict(self, idx):

        data_dict_path = os.path.join(self.scenario_path, 'meta', f'{idx:04d}_data_dict.pth')
        data_dict = torch.load(data_dict_path)
        # data_dict = {}
        # text_input_question = ""
        # text_input_answer = ""
        # img_path = os.path.join(scenario_path, 'rgb_front', f'{idx:04d}.png')
        # meta_path = os.path.join(scenario_path,'meta', f'{idx:04d}.json')
        # print('meta_path', meta_path)

        # # load image
        # img_tensor_path = os.path.join(scenario_path, 'meta', f'{idx:04d}_image.pt')
        # img_tensor = torch.load(img_tensor_path)
        # img = Image.open(img_path)
        # img = np.array(img)
        # image, image_size = self.prepare_clip_data(img)
        # data_dict['image'] = [img_tensor]
        # data_dict['image_sizes'] = image_size

        # # action data
        # ego_state_tensor_path = os.path.join(scenario_path, 'meta', f'{idx:04d}_ego_state.pt')
        # ego_state_tensor = torch.load(ego_state_tensor_path)
        # ego_state, command = self.prepare_action_data(meta_path)
        # data_dict['ego_state'] = ego_state_tensor
        # data_dict['command'] = command

        # # navigation data
        # navi_tensor_path = os.path.join(scenario_path, 'meta', f'{idx:04d}_navi.pt')
        # navi_tensor = torch.load(navi_tensor_path)
        # navi_path, navi_mask = self.prepare_navigation_data(meta_path)
        # data_dict['navi_path'] = navi_tensor
        # data_dict['navi_mask'] = navi_mask

        # # text data
        # text_input_answer += TEXT_ANSWER_EGO_TRAJ
        # text_input_question = TEXT_INPUT_HEAD + text_input_question
        # text_question = TEXT_PROMPT + text_input_question + TEXT_TASK
        # texts = [{'from': 'Question', 'value': text_question}, {'from': 'Answer', 'value': text_input_answer}]
        # data_dict.update(preprocess(texts, self.tokenizer))


        return data_dict
    
    def prepare_navigation_data(self, meta_path):
        meta = json.load(open(meta_path))
        navi_path = torch.tensor(meta['navigation_points'])
        navi_mask = torch.tensor([1] * len(navi_path))

        return navi_path, navi_mask

    def prepare_action_data(self, meta_path):
        meta = json.load(open(meta_path))
        ego_state = meta['history_states']
        command = meta['command']
        return torch.tensor(ego_state), torch.tensor(command)

    def prepare_clip_data(self, img):
        image = Image.fromarray(img).convert('RGB')
        processor = self.cfg.image_processor
        image_size = [image.size]
        if self.cfg.image_aspect_ratio == 'anyres':
            image = process_images([image], processor, self.cfg)
        else:
            raise ValueError(f"Invalid image aspect ratio: {self.cfg.image_aspect_ratio}")
        return image, image_size




model_path = '/data/huazh/Bench2Drive/LLava-Next-Nuscenes/checkpoints/encoder-query-norm-dropout0.4/checkpoint-22014'
model_name = 'llava_qwen_1_5-0.5B'
torch_dtype = 'bf16'
attn_implementation="sdpa"
scenario_path = '/data42/huazh/Bench2Drive/Bench2Drive/eval_output/Dev10_padding0.5/checkpoint-25683/Dev10_RouteScenario_26405_rep0_Town15_StaticCutIn_1_0_11_19_14_01_08'

agent = AblationAgent(model_path, model_name, torch_dtype, attn_implementation, scenario_path)

idx = 63

'''
1. 确定block场景idx
2. 更改ego state, 测试输出
3. 更改navigation, 测试输出
'''

# result = []
# for i in range(100):
#     pred_traj = agent.inference(idx)
#     # agent.ablation_navigation(idx)
#     print('pred_traj: \n', pred_traj)
#     result.append(pred_traj)

# result = np.array(result)
# print('result: \n', np.mean(result, axis=0))

agent.create_video()
# path = '/data42/huazh/Bench2Drive/Bench2Drive/eval_output/Dev10_padding0.5/checkpoint-25683'
# scenarios = os.listdir(path)

# for scenario in scenarios:
#     scenario_path = os.path.join(path, scenario)
#     # 判断是否为文件夹
#     if not os.path.isdir(scenario_path):
#         continue
#     print(f"Scenario: {scenario_path}")
#     agent = AblationAgent(model_path, model_name, torch_dtype, attn_implementation, scenario_path)
#     agent.create_video()



