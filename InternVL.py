"""
属于 InternVL 2.5系列
视频理解与生成：可以用于视频内容的分析、总结和生成相关的文本描述。
视觉问答：能够回答与图像或视频内容相关的问题。
多模态对话：支持与用户进行包含视觉信息的对话。
"""

# 导入必要的库
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
from modelscope.utils.constant import Tasks

# ========== 关键修改1：模型配置（适配ModelScope的safetensors格式） ==========
MODEL_PATH = "/root/autodl-tmp/InternVideo2_5_Chat_8B/"
# 强制指定torch dtype，避免精度冲突
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ========== 关键修改2：初始化分词器和模型（适配最新版InternVL 2.5） ==========
# 加载分词器（显式指定trust_remote_code=True，加载模型自定义代码）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True,
    revision="master"  # 锁定模型版本，避免自动拉取不兼容版本
)

# 加载模型（移除冲突的half()，统一用torch_dtype，适配safetensors）
model = AutoModel.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True,
    revision="master",
    torch_dtype=torch_dtype,
    device_map="auto"  # 自动分配GPU/CPU，避免手动cuda()的冲突
)
# 模型移到GPU并设置eval模式（避免训练模式的额外参数）
model = model.eval().cuda()

# ImageNet 数据集的均值和标准差
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    构建图像转换pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    寻找最接近原始图像宽高比的目标比例
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    动态预处理图像，根据宽高比将图像分割成多个块
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算现有图像宽高比
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 寻找最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, 
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=6):
    """
    加载并处理图像
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    获取视频帧索引
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    """
    根据视频时长计算帧数
    """
    local_num_frames = 4        
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    """
    加载并处理视频
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# ========== 关键修改3：简化generation_config（适配InternVL 2.5的生成逻辑） ==========
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.9,  # 原0.1过小，易导致生成空内容，调整为0.9
    num_beams=1,
    pad_token_id=tokenizer.pad_token_id,  # 显式指定pad_token，避免配置缺失
    eos_token_id=tokenizer.eos_token_id   # 显式指定eos_token
)

# 视频路径（确保路径正确）
VIDEO_PATH = "/root/autodl-tmp/Video/car.mp4"
num_segments=128

# ========== 关键修改4：torch.no_grad()包裹，且调整pixel_values的dtype ==========
with torch.no_grad():
  # 加载视频并处理
  pixel_values, num_patches_list = load_video(
      VIDEO_PATH, 
      num_segments=num_segments, 
      max_num=1, 
      get_frame_by_duration=False
  )
  # 对齐pixel_values和模型的dtype，避免类型冲突
  pixel_values = pixel_values.to(torch_dtype).to(model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  
  # 单轮对话：视频详细描述（英文）
  question1 = "Describe this video in detail."
  question = video_prefix + question1
  output1, chat_history = model.chat(
      tokenizer, 
      pixel_values, 
      question, 
      generation_config=generation_config, 
      num_patches_list=num_patches_list, 
      history=None, 
      return_history=True
  )
  print("=== 视频详细描述（英文）===")
  print(output1)
  
  # 多轮对话：询问视频中的人数（英文）
  question2 = "How many people appear in the video?"
  output2, chat_history = model.chat(
      tokenizer, 
      pixel_values, 
      question2, 
      generation_config=generation_config, 
      num_patches_list=num_patches_list, 
      history=chat_history, 
      return_history=True
  )
  print("\n=== 视频中人数 ===")
  print(output2)

  # 中文对话：询问车辆损伤部位
  question3 = "车的哪个部位损伤了？"
  question_cn = video_prefix + question3
  output3, chat_history = model.chat(
      tokenizer, 
      pixel_values, 
      question_cn, 
      generation_config=generation_config, 
      num_patches_list=num_patches_list, 
      history=None, 
      return_history=True
  )
  print("\n=== 车辆损伤部位 ===")
  print(output3)
  
  # 中文多轮：询问碰撞位置
  question4 = "车撞到哪里了？"
  output4, chat_history = model.chat(
      tokenizer, 
      pixel_values, 
      question4, 
      generation_config=generation_config, 
      num_patches_list=num_patches_list, 
      history=chat_history, 
      return_history=True
  )
  print("\n=== 车辆碰撞位置 ===")
  print(output4)