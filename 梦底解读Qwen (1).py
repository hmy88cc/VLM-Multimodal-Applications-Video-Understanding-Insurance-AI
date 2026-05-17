"""
适配2026春晚《梦底》视频解读 - Qwen3.6-27B-FP8
优化版（新增基础信息识别+保留原有深度分析）：
1. 48帧采样（含前3秒关键帧），显存占用≈24GB（5090 32GB剩余8GB）
2. 新增基础信息解读（节目名/词曲/演唱/伴舞），强制关注前3秒左下角红色字体
3. 保留演唱/舞蹈/视觉/春晚特色深度分析
4. 每轮对话清空显存，避免OOM
5. 提问强制要求结合具体细节（时间段/歌词/动作）
"""

# 导入必要的库
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
import gc

# ========== 显存强优化：限额+清空缓存 ==========
torch.cuda.empty_cache()
gc.collect()
# 限制进程使用90%显存，预留余量
torch.cuda.set_per_process_memory_fraction(0.9, device=0)

# 模型配置（替换为你的实际模型路径）
MODEL_PATH = "/root/autodl-tmp/Qwen3.6-27B-FP8/"
model_path = MODEL_PATH

# 初始化分词器和模型（极致显存优化）
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 仅用bfloat16，禁用不必要的精度，device_map严格限定GPU
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
# 先加载配置，手动禁用量化
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# 删除量化配置，以原始精度加载
if hasattr(config, 'quantization_config'):
    config.quantization_config = None

model = AutoModel.from_pretrained(
    model_path,
    config=config,  # 使用修改后的配置
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map={"": 0},  # 强制所有层加载到GPU 0，避免分散占用
    low_cpu_mem_usage=True,  # 添加低CPU内存占用参数
).eval()
# 手动移到GPU，避免device_map自动分配的显存碎片
model = model.cuda()

# ImageNet 均值/标准差（固定）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """构建图像转换pipeline（适配高分辨率视频）"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """寻找最接近的宽高比，适配2536×1700分辨率"""
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

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    """动态预处理：max_num=1，避免分块过多爆显存"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, 
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=1):
    """加载单帧图像（适配max_num=1）"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=48):
    """
    核心优化：强制包含前3秒关键帧（0-7帧，fps=24，3秒=72帧，取前8帧确保覆盖细节）
    """
    # 强制采样前3秒的关键帧（保证细节识别）
    start_3s_frames = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    
    # 剩余帧数：48 - 8 = 40帧，均匀采样
    remaining_segments = num_segments - len(start_3s_frames)
    seg_size = float(end_idx - start_idx) / remaining_segments
    remaining_frames = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(remaining_segments)])
    
    # 合并+去重+排序，确保总数48帧
    frame_indices = np.unique(np.concatenate([start_3s_frames, remaining_frames]))
    if len(frame_indices) < num_segments:
        padding_frames = np.arange(max(frame_indices)+1, max(frame_indices)+1 + (num_segments - len(frame_indices)))
        frame_indices = np.concatenate([frame_indices, padding_frames])[:num_segments]
    
    return frame_indices

def get_num_frames_by_duration(duration):
    """固定48帧，禁用时长计算"""
    return 48

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=48, get_frame_by_duration = False):
    """加载春晚视频，48帧采样（含前3秒关键帧）"""
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
        frame_index = min(frame_index, max_frame)  # 防止越界
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# 生成配置：显存友好+输出精准
generation_config = dict(
    do_sample=True,
    temperature=0.1,  # 低温度保证准确性
    max_new_tokens=2048,  # 足够详细且不超显存
    top_p=0.9,
    num_beams=1,  # 关闭束搜索，大幅降低显存占用
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1  # 避免重复套话
)

# ========== 你的视频配置 ==========
VIDEO_PATH = "/root/autodl-tmp/Video/春晚梦底.mp4"  # 替换为实际路径
video_path = VIDEO_PATH
num_segments=48  # 核心：48帧，显存安全
input_size=448
max_num=1

# ========== 核心解读逻辑（每轮清空显存） ==========
with torch.no_grad():
  # 加载视频（48帧采样，含前3秒关键帧）
  pixel_values, num_patches_list = load_video(
      video_path, 
      num_segments=num_segments, 
      max_num=max_num, 
      input_size=input_size,
      get_frame_by_duration=False
  )
  pixel_values = pixel_values.to(torch_dtype).to(model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  chat_history = None  # 初始化对话历史
  
  # ===== 0. 视频基础信息解读（新增：优先解读，重点关注前3秒左下角红色字体） =====
  print("===== 视频基础信息解读 =====")
  q_base_info = """请严格按照以下要求提取视频基础信息，必须重点查看视频前3秒画面的左下角红色字体：
  1. 节目完整名称：准确写出完整节目名；
  2. 词曲作者：明确写出该歌曲的作词人和作曲人；
  3. 核心演唱者：明确写出主要演唱人员；
  4. 主要伴舞：明确写出主要伴舞人员；
  要求：所有信息必须基于视频前3秒左下角红色字体内容，不得编造，不确定的信息标注「未识别」。"""
  output_base_info, chat_history = model.chat(
      tokenizer, pixel_values, video_prefix + q_base_info, 
      generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True
  )
  print(f"问题：{q_base_info}\n回答：{output_base_info}\n")
  # 清空本轮显存
  torch.cuda.empty_cache()
  gc.collect()

  # ===== 1. 海来阿木演唱深度分析（强制结合细节） =====
  print("===== 海来阿木演唱深度分析 =====")
  q_sing = """请从以下维度详细评价海来阿木的演唱，必须结合具体细节：
  1. 音准和气息控制：举例说明在《梦底》的哪句歌词处体现了气息稳定/音准精准？有无失误？
  2. 情感表达：他的唱腔（如烟嗓/转音）在哪个歌词段落传递了「梦」/「思念」的主题？
  3. 演唱风格：具体是哪种风格（如彝族民族风+流行抒情）？举例说明高低音处理的细节（如某句高音的音色）？
  4. 与歌曲适配度：他的嗓音特点（如磁性/沙哑）如何契合《梦底》的旋律节奏？"""
  output_sing, chat_history = model.chat(
      tokenizer, pixel_values, video_prefix + q_sing, 
      generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True
  )
  print(f"问题：{q_sing}\n回答：{output_sing}\n")
  # 清空本轮显存
  torch.cuda.empty_cache()
  gc.collect()

  # ===== 2. 刘浩存舞蹈深度分析（强制结合细节） =====
  print("===== 刘浩存舞蹈深度分析 =====")
  q_dance = """请从以下维度详细分析刘浩存的舞蹈，必须结合具体画面细节：
  1. 舞蹈功底：举例说明她在视频第XX秒的肢体动作（如手臂舒展/旋转）体现了协调性/流畅度？有无失误？
  2. 标志性动作：具体描述1-2个核心动作（如手部兰花指/脚步碎步），说明出现在视频的哪个时间段？
  3. 动作寓意：该动作如何对应《梦底》的具体歌词（如“梦底的思念”）？
  4. 歌舞配合：她在歌曲慢板（如第XX秒）的柔缓动作/快板（如第XX秒）的卡点动作具体是什么？"""
  output_dance, chat_history = model.chat(
      tokenizer, pixel_values, video_prefix + q_dance, 
      generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True
  )
  print(f"问题：{q_dance}\n回答：{output_dance}\n")
  # 清空本轮显存
  torch.cuda.empty_cache()
  gc.collect()

  # ===== 3. 视觉细节精准解读（强制结合细节） =====
  print("===== 视觉细节精准解读 =====")
  q2_cn = """请详细描述以下内容，必须结合具体画面：
  1. 服装：海来阿木的服装是否有红色春晚元素（如刺绣/配饰）？刘浩存的长裙具体颜色（如酒红/玫红）、材质（如真丝/纱质）？
  2. 舞台：背景屏幕的具体画面（如星空/山水）？灯光的颜色（如中国红/暖黄）在哪个时间段变化？
  3. 契合度：服装/舞台的红色元素如何契合春晚+《梦底》的主题？"""
  output2, chat_history = model.chat(
      tokenizer, pixel_values, video_prefix + q2_cn, 
      generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True
  )
  print(f"问题：{q2_cn}\n回答：{output2}\n")
  # 清空本轮显存
  torch.cuda.empty_cache()
  gc.collect()

  # ===== 4. 春晚特色精准解读（最后一轮，显存预留充足） =====
  print("===== 春晚特色精准解读 =====")
  q4_cn = """请具体说明以下内容，必须结合视频画面：
  1. 春晚标志性元素：除了红色主色调，是否有主持人串词/观众席新年装饰/倒计时元素？出现在视频哪个时间段？
  2. 氛围契合度：该表演的「温情/团圆」氛围如何通过舞台/服装/表演体现？举例说明。"""
  output4, chat_history = model.chat(
      tokenizer, pixel_values, video_prefix + q4_cn, 
      generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True
  )
  print(f"问题：{q4_cn}\n回答：{output4}\n")

# 最终清空所有显存
torch.cuda.empty_cache()
gc.collect()
print("===== 解读完成，显存已清空 =====")