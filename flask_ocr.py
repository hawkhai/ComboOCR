from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import torch
import numpy as np
import base64
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import time
import uuid
import sys
import textwrap
import gc

# 导入新的OCR和工具函数
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img
from utils.utils_doctr_plus import *
from utils.utils_gcdrnet import *
from model.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 配置上传文件夹
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 全局变量声明
models = None
device = None

def sort_ocr_results(ocr_result):
    """
    将OCR结果按照从上到下、从左到右的顺序排序
    垂直方向优先级高于水平方向，使用固定阈值区分行

    Args:
        ocr_result: OCR识别结果列表，每个元素包含polygon和text

    Returns:
        排序后的文本字符串
    """
    # 检查OCR结果是否为空
    if not ocr_result:
        return ""

    # 计算每个文本框的中心点
    for item in ocr_result:
        polygon = item['polygon']

        # 计算中心点
        center_x = sum(polygon[0::2]) / 4  # x坐标
        center_y = sum(polygon[1::2]) / 4  # y坐标

        # 添加中心点坐标到item
        item['center_x'] = center_x
        item['center_y'] = center_y

    # 计算所有文本框的高度
    heights = []
    for item in ocr_result:
        polygon = item['polygon']
        y_coords = polygon[1::2]
        height = max(y_coords) - min(y_coords)
        heights.append(height)

    # 使用平均高度的一半作为行分组阈值
    avg_height = sum(heights) / max(1, len(heights))
    line_threshold = avg_height / 2

    # 按y坐标排序
    sorted_by_y = sorted(ocr_result, key=lambda x: x['center_y'])

    # 使用简单阈值进行行分组
    lines = []
    if sorted_by_y:
        current_line = [sorted_by_y[0]]
        base_y = sorted_by_y[0]['center_y']

        for item in sorted_by_y[1:]:
            # 检查是否应该开始新行
            if abs(item['center_y'] - base_y) > line_threshold:
                # 对当前行按x坐标从左到右排序
                current_line = sorted(current_line, key=lambda x: x['center_x'])
                lines.append(current_line)

                # 开始新行
                current_line = [item]
                base_y = item['center_y']
            else:
                # 添加到当前行
                current_line.append(item)

        # 添加最后一行（并排序）
        if current_line:
            current_line = sorted(current_line, key=lambda x: x['center_x'])
            lines.append(current_line)

    # 拼接文本
    result_text = ""
    for line in lines:
        line_text = "".join(item['text'] for item in line)
        result_text += line_text + "\n"

    return result_text.strip()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_models(device_name):
    """初始化模型"""
    print("正在加载模型...")
    device = torch.device(device_name)

    try:
        # ONNX OCR模型 (检测+识别一体) - 必须加载
        ocr_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
        print("ONNX OCR 模型加载成功")

        # 文本图像扭曲矫正模型(DocTr++) - 必须加载
        dewarp_path = "./models/doctr_plus/model.pt"
        if not os.path.exists(dewarp_path):
            raise FileNotFoundError(f"dewarp模型文件不存在: {dewarp_path}")

        dewarp_model = DocTr_Plus(weights=dewarp_path, device=device)
        print("扭曲矫正模型加载成功")

        # 文本图像外观增强模型(GCDRNet) - 可选加载
        gcnet = None
        drnet = None
        try:
            gcnet_path = './models/gcdr_net/gcnet/checkpoint.pkl'
            drnet_path = './models/gcdr_net/drnet/checkpoint.pkl'

            if os.path.exists(gcnet_path) and os.path.exists(drnet_path):
                gcnet = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).to(device)
                state = convert_state_dict(
                    torch.load(gcnet_path, map_location=device)['model_state'])
                gcnet.load_state_dict(state)
                gcnet.eval()
                print("gcnet模型加载成功")

                drnet = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).to(device)
                state = convert_state_dict(
                    torch.load(drnet_path, map_location=device)['model_state'])
                drnet.load_state_dict(state)
                drnet.eval()
                print("drnet模型加载成功")
            else:
                print("外观增强模型文件未找到，将跳过外观增强功能")

        except Exception as e:
            print(f"外观增强模型加载失败: {str(e)}")
            print("将继续运行，但外观增强功能不可用")
            gcnet = None
            drnet = None

        print("模型初始化完成！")
        return {
                   'ocr_model': ocr_model,
                   'gcnet': gcnet,  # 可能为None
                   'drnet': drnet,  # 可能为None
                   'dewarp_model': dewarp_model
               }, device

    except Exception as e:
        print(f"关键模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def convert_paddleocr_to_standard_format(paddle_result):
    """
    将PaddleOCR的结果格式转换为标准格式

    PaddleOCR格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)]
    标准格式: {'polygon': [x1,y1,x2,y2,x3,y3,x4,y4], 'text': 'text'}
    """
    ocr_result = []

    if not paddle_result or not paddle_result[0]:
        return ocr_result

    for item in paddle_result[0]:
        # 提取坐标点和文本
        points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text_info = item[1]  # ('text', confidence)
        text = text_info[0]

        # 将坐标转换为扁平列表格式 [x1,y1,x2,y2,x3,y3,x4,y4]
        polygon = []
        for point in points:
            polygon.extend([float(point[0]), float(point[1])])

        ocr_result.append({
            'polygon': polygon,
            'text': text
        })

    return ocr_result

def apply_appearance_enhancement(image, models, device):
    """
    应用外观增强处理

    Args:
        image: 输入图像
        models: 模型字典
        device: 设备

    Returns:
        enhanced_image: 增强后的图像
    """
    if models['gcnet'] is None or models['drnet'] is None:
        print("外观增强模型未加载，跳过外观增强步骤")
        return image

    # 外观增强处理
    im_org, padding_h, padding_w = stride_integral(image)
    h, w = im_org.shape[:2]
    im = im_org

    # 预先声明变量，便于清理
    im_tensor = None
    im_org_tensor = None
    shadow = None
    model1_im = None
    pred = None

    try:
        with torch.no_grad():
            # 创建张量
            im_tensor = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0).float().to(device)
            im_org_tensor = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0).float().to(device)

            # 第一个模型推理
            shadow = models['gcnet'](im_tensor)
            shadow = F.interpolate(shadow, (h, w))

            # 第二个模型推理
            model1_im = torch.clamp(im_org_tensor / shadow, 0, 1)
            pred, _, _, _ = models['drnet'](torch.cat((im_org_tensor, model1_im), 1))

            # 转换为 numpy，立即移动到 CPU
            shadow_np = shadow[0].permute(1, 2, 0).detach().cpu().numpy()
            shadow_np = (shadow_np * 255).astype(np.uint8)
            shadow_np = shadow_np[padding_h:, padding_w:]

            model1_im_np = model1_im[0].permute(1, 2, 0).detach().cpu().numpy()
            model1_im_np = (model1_im_np * 255).astype(np.uint8)
            model1_im_np = model1_im_np[padding_h:, padding_w:]

            pred_np = pred[0].permute(1, 2, 0).detach().cpu().numpy()
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_np = pred_np[padding_h:, padding_w:]

    finally:
        # 显式删除GPU张量
        for tensor in [im_tensor, im_org_tensor, shadow, model1_im, pred]:
            if tensor is not None:
                del tensor
        # 强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 触发垃圾回收
        gc.collect()
    enhanced_image = np.clip(pred_np, 0, 255).astype(np.uint8)
    return enhanced_image

def apply_dewarp_correction(image, models, device):
    """
    应用扭曲矫正处理

    Args:
        image: 输入图像
        models: 模型字典
        device: 设备

    Returns:
        corrected_image: 矫正后的图像
    """
    # 文本图像扭曲矫正
    img_ori = image / 255.
    h_, w_, c_ = img_ori.shape
    img_ori = cv2.resize(img_ori, (2560, 2560))
    h, w, _ = img_ori.shape
    img = cv2.resize(img_ori, (288, 288))
    img = img.transpose(2, 0, 1)
    
    # 预先声明变量
    img_tensor = None
    bm = None
    
    try:
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)

        with torch.no_grad():
            bm = models['dewarp_model'](img_tensor)
            bm_np = bm.detach().cpu().numpy()[0]  # 立即移动到 CPU
            
        bm0 = bm_np[0, :, :]
        bm1 = bm_np[1, :, :]
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))

        img_geo = cv2.remap(img_ori, bm0, bm1, cv2.INTER_LINEAR) * 255
        img_geo = cv2.resize(img_geo, (w_, h_))
    finally:
        # 显式删除GPU张量  
        for tensor in [img_tensor, bm]:
            if tensor is not None:
                del tensor  
        # 强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 触发垃圾回收
        gc.collect()
    # 确保图像是uint8类型
    img_geo = np.clip(img_geo, 0, 255).astype(np.uint8)
    return img_geo

def create_text_image_with_boxes(ocr_result, width, height, font_size=100):
    """
    创建显示文本的图像，文本根据对应box位置显示，并用相同颜色框出

    Args:
        ocr_result: OCR结果列表，包含polygon和text
        width: 图像宽度
        height: 图像高度
        font_size: 初始字体大小

    Returns:
        文本图像（numpy数组）
    """
    # 创建白色背景图像
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # 如果没有OCR结果，返回提示图像
    if not ocr_result:
        try:
            font = ImageFont.truetype("./onnxocr/fonts/simfang.ttf", font_size)
        except:
            font = ImageFont.load_default()

        msg = "未识别到文本"
        text_bbox = draw.textbbox((0, 0), msg, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), msg, fill='gray', font=font)
        return np.array(img)

    # 定义与检测框相同的颜色
    colors = [
        (255, 99, 71),  # 番茄红
        (60, 179, 113),  # 海洋绿
        (30, 144, 255),  # 道奇蓝
        (255, 165, 0),  # 橙色
        (218, 112, 214),  # 兰花紫
        (32, 178, 170),  # 浅海洋绿
        (255, 20, 147),  # 深粉红
        (0, 191, 255),  # 深天蓝
        (50, 205, 50),  # 酸橙绿
        (255, 215, 0),  # 金色
    ]

    # 为每个文本创建显示区域
    text_regions = []

    for i, item in enumerate(ocr_result):
        polygon = item['polygon']
        text = item['text']
        color = colors[i % len(colors)]

        if not text or not text.strip():
            continue

        # 计算text box的边界矩形
        x_coords = polygon[0::2]  # x坐标
        y_coords = polygon[1::2]  # y坐标

        min_x = max(0, int(min(x_coords)))
        max_x = min(width, int(max(x_coords)))
        min_y = max(0, int(min(y_coords)))
        max_y = min(height, int(max(y_coords)))

        # 计算区域尺寸
        region_width = max_x - min_x
        region_height = max_y - min_y

        if region_width <= 10 or region_height <= 10:
            continue

        text_regions.append({
            'text': text.strip(),
            'bbox': (min_x, min_y, max_x, max_y),
            'color': color,
            'index': i + 1,
            'region_width': region_width,
            'region_height': region_height
        })

    # 为每个文本区域单独优化字体大小
    for region in text_regions:
        text = region['text']
        min_x, min_y, max_x, max_y = region['bbox']
        color = region['color']
        index = region['index']

        # 计算可用空间，使用更小的边距
        margin = max(1, min(3, min(region['region_width'], region['region_height']) // 15))  # 减小边距
        available_width = max_x - min_x - margin * 2
        available_height = max_y - min_y - margin * 2

        if available_width <= 3 or available_height <= 3:
            continue

        # 为当前文本找到最适合的字体大小 - 更激进的缩小策略
        current_font_size = min(font_size, available_height - 2)
        min_font_size = 12  # 降低最小字体大小
        best_font_size = min_font_size
        display_text = text

        # 首先尝试找到能显示完整文本的最大字体
        while current_font_size >= min_font_size:
            try:
                test_font = ImageFont.truetype("./onnxocr/fonts/simfang.ttf", current_font_size)
            except:
                test_font = ImageFont.load_default()

            # 测试文本是否能放入区域
            text_bbox = draw.textbbox((0, 0), text, font=test_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 如果文本能完全适合，选择这个字体大小
            if text_width <= available_width * 1.2 and text_height <= available_height:  # 允许稍微超出20%
                best_font_size = current_font_size
                display_text = text  # 使用完整文本
                break

            current_font_size -= 1

        # 如果即使最小字体也放不下完整文本，则考虑截断
        if current_font_size < min_font_size:
            # 使用最小字体重新计算
            try:
                final_font = ImageFont.truetype("./onnxocr/fonts/simfang.ttf", min_font_size)
            except:
                final_font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), text, font=final_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            best_font_size = min_font_size

            # 如果文本宽度超出太多，才进行截断
            if text_width > available_width * 1.5:  # 提高截断阈值
                # 计算能显示多少字符
                char_width = text_width / len(text)
                max_chars = int((available_width * 1.3) / char_width) - 2  # 为省略号留更少空间
                
                if max_chars > len(text) * 0.7:  # 如果能显示70%以上的文字，就不截断
                    display_text = text
                elif max_chars > 3:
                    display_text = text[:max_chars] + ".."  # 使用两个点而不是三个
                else:
                    display_text = text[:2] + ".." if len(text) > 2 else text
            else:
                display_text = text  # 即使稍微超出也显示完整文本

        # 使用最终确定的字体大小
        try:
            final_font = ImageFont.truetype("./onnxocr/fonts/simfang.ttf", best_font_size)
        except:
            final_font = ImageFont.load_default()

        # 计算最终文本尺寸
        final_text_bbox = draw.textbbox((0, 0), display_text, font=final_font)
        final_text_width = final_text_bbox[2] - final_text_bbox[0]
        final_text_height = final_text_bbox[3] - final_text_bbox[1]

        # 计算文本位置（居中，但允许稍微超出边界）
        text_x = min_x + margin + max(0, (available_width - final_text_width) // 2)
        text_y = min_y + margin + max(0, (available_height - final_text_height) // 2)

        # 确保文本不会严重超出边界
        text_x = min(text_x, max_x - margin)  # 允许文本稍微超出右边界
        text_y = min(text_y, max_y - final_text_height - margin)
        text_x = max(text_x, min_x + margin)
        text_y = max(text_y, min_y + margin)

        # 动态调整背景框大小以适应文本
        padding = 1
        bg_x1 = max(0, min(min_x, text_x - padding))
        bg_y1 = max(0, min(min_y, text_y - padding))
        bg_x2 = min(width, max(max_x, text_x + final_text_width + padding))
        bg_y2 = min(height, max(max_y, text_y + final_text_height + padding))

        # 创建半透明背景
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 背景透明度根据是否截断调整
        alpha = 80 if display_text == text else 100  # 完整文本时背景更透明
        overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2],
                               fill=(*color, alpha), outline=color, width=1)

        # 将overlay合成到主图像
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)  # 重新创建draw对象

        # 绘制文本
        draw.text((text_x, text_y), display_text, fill='black', font=final_font)

        # 在左上角绘制序号（使用较小字体）
        try:
            number_font = ImageFont.truetype("./onnxocr/fonts/simfang.ttf", max(8, best_font_size // 2))
        except:
            number_font = ImageFont.load_default()

        number_x = bg_x1 + 1
        number_y = bg_y1 + 1
        # 绘制序号背景
        num_bbox = draw.textbbox((0, 0), str(index), font=number_font)
        num_width = num_bbox[2] - num_bbox[0]
        num_height = num_bbox[3] - num_bbox[1]
        draw.rectangle([number_x - 1, number_y - 1, number_x + num_width + 1, number_y + num_height + 1],
                       fill=color)
        draw.text((number_x, number_y), str(index), fill='white', font=number_font)

    return np.array(img)

def draw_text_boxes(image, ocr_result):
    """
    在图像上绘制OCR检测的文本框

    Args:
        image: 输入图像
        ocr_result: OCR结果列表

    Returns:
        绘制了文本框的图像
    """
    if not ocr_result:
        return image

    # 复制图像以避免修改原图
    result_image = image.copy()

    # 定义一组漂亮的颜色
    colors = [
        (255, 99, 71),  # 番茄红
        (60, 179, 113),  # 海洋绿
        (30, 144, 255),  # 道奇蓝
        (255, 165, 0),  # 橙色
        (218, 112, 214),  # 兰花紫
        (32, 178, 170),  # 浅海洋绿
        (255, 20, 147),  # 深粉红
        (0, 191, 255),  # 深天蓝
        (50, 205, 50),  # 酸橙绿
        (255, 215, 0),  # 金色
    ]

    for i, item in enumerate(ocr_result):
        polygon = item['polygon']
        text = item['text']

        # 选择颜色
        color = colors[i % len(colors)]

        # 将polygon转换为点坐标
        points = []
        for j in range(0, len(polygon), 2):
            points.append((int(polygon[j]), int(polygon[j + 1])))

        # 绘制多边形框
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_image, [pts], True, color, 2)
    return result_image

def process_image(image_path, models, device, use_enhancement=False, use_dewarp=False):
    """
    处理单张图片并返回OCR结果

    Args:
        image_path: 图像路径
        models: 模型字典
        device: 设备
        use_enhancement: 是否使用外观增强
        use_dewarp: 是否使用扭曲矫正

    Returns:
        结果字典
    """
    # 验证模型是否正确加载
    if models is None:
        raise ValueError("模型未正确初始化")

    required_models = ['ocr_model', 'dewarp_model']
    for model_name in required_models:
        if model_name not in models or models[model_name] is None:
            raise ValueError(f"关键模型 {model_name} 未正确加载")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    print(f"开始处理图像: {os.path.basename(image_path)}")
    print(f"   - 外观增强: {'启用' if use_enhancement else '禁用'}")
    print(f"   - 扭曲矫正: {'启用' if use_dewarp else '禁用'}")

    try:
        # 存储原始图像
        processed_images = {}
        _, img_encoded = cv2.imencode('.png', image)
        processed_images['original'] = base64.b64encode(img_encoded).decode('utf-8')

        # 1. 可选：外观增强
        if use_enhancement:
            image = apply_appearance_enhancement(image, models, device)
        else:
            print("跳过外观增强步骤")

        # 2. 可选：文本图像扭曲矫正
        if use_dewarp:
            image = apply_dewarp_correction(image, models, device)
        else:
            print("跳过扭曲矫正步骤")

        # 保存处理后的图像用于可视化
        processed_image = image.copy()

        # 3. ONNX OCR推理（检测+识别一体）
        paddle_result = models['ocr_model'].ocr(image)

        # 4. 转换结果格式
        ocr_result = convert_paddleocr_to_standard_format(paddle_result)

        # 5. 处理OCR结果，排序拼接
        if not ocr_result:
            sorted_text = ""
        else:
            sorted_text = sort_ocr_results(ocr_result)

        # 6. 创建可视化图像
        print("🎨 正在创建可视化图像...")
        # 在处理后的图像上绘制文本框
        image_with_boxes = draw_text_boxes(processed_image, ocr_result)
        # 创建显示ocr_result的图像
        img_height, img_width = image_with_boxes.shape[:2]
        text_image = create_text_image_with_boxes(ocr_result, img_width, img_height)
        # 将两张图像水平拼接
        visualization = np.hstack([image_with_boxes, text_image])
        # 将可视化图像编码为base64
        _, img_encoded = cv2.imencode('.png', visualization)
        visualization_base64 = base64.b64encode(img_encoded).decode('utf-8')
        print("可视化图像创建完成")

        # 准备处理信息
        processing_info = {
            'use_enhancement': use_enhancement,
            'use_dewarp': use_dewarp,
            'text_regions_count': len(ocr_result) if ocr_result else 0
        }

        return {
            'sorted_text': sorted_text,
            'ocr_result': ocr_result,
            'visualization': visualization_base64,
            'processing_info': processing_info
        }
        
    finally:
        # 在函数结束时清理内存
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        gc.collect()

@app.route('/process', methods=['POST'])
def process():
    # 检查是否有文件部分
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # 获取处理参数
        use_enhancement = request.form.get('use_enhancement', 'false').lower() == 'true'
        use_dewarp = request.form.get('use_dewarp', 'false').lower() == 'true'

        # 检查外观增强功能是否可用
        if use_enhancement and (models['gcnet'] is None or models['drnet'] is None):
            return jsonify({"error": "外观增强模型未加载，无法使用此功能"}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        # 添加时间戳和随机字符串，避免文件名冲突
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{unique_id}{ext}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # 处理图像
        result = process_image(
            filepath,
            models,
            device,
            use_enhancement=use_enhancement,
            use_dewarp=use_dewarp
        )
        # print(f"处理结果: {result['sorted_text']}")

        # 清理临时文件
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Health check endpoint removed to prevent connection issues

@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR文本识别系统</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                line-height: 1.6;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                padding: 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="rgba(255,255,255,0.1)"><polygon points="0,0 1000,0 1000,100 0,80"/></svg>');
                background-size: cover;
            }

            .header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 10px;
                position: relative;
                z-index: 1;
            }

            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }

            .content {
                padding: 40px;
            }

            .form-section {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05);
                border: 1px solid rgba(229, 231, 235, 0.8);
                transition: all 0.3s ease;
            }

            .form-section:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.1);
            }

            .form-group {
                margin-bottom: 25px;
            }

            .form-group label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #374151;
                font-size: 1.1rem;
            }

            .file-input-wrapper {
                position: relative;
                display: inline-block;
                width: 100%;
            }

            .file-input {
                position: absolute;
                left: -9999px;
                opacity: 0;
            }

            .file-input-label {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                width: 100%;
                padding: 20px;
                border: 3px dashed #d1d5db;
                border-radius: 12px;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 1.1rem;
                font-weight: 500;
                color: #6b7280;
            }

            .file-input-label:hover {
                border-color: #4f46e5;
                background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
                color: #4f46e5;
                transform: translateY(-1px);
            }

            .file-input-label.has-file {
                border-color: #10b981;
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                color: #059669;
            }

            .upload-icon {
                font-size: 1.5rem;
            }

            .checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
                padding: 25px;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }

            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 15px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .checkbox-item:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .checkbox-item input[type="checkbox"] {
                width: 20px;
                height: 20px;
                accent-color: #4f46e5;
                cursor: pointer;
            }

            .checkbox-item label {
                margin: 0;
                font-weight: 500;
                cursor: pointer;
                color: #374151;
            }

            .checkbox-item.disabled {
                opacity: 0.5;
                pointer-events: none;
                background: #f9fafb;
            }

            .process-btn {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                border: none;
                padding: 16px 32px;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
                position: relative;
                overflow: hidden;
            }

            .process-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6);
            }

            .process-btn:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            .process-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }

            .process-btn:hover::before {
                left: 100%;
            }

            #loading {
                display: none;
                margin: 30px 0;
                text-align: center;
                padding: 40px;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border-radius: 16px;
                border: 1px solid #e5e7eb;
            }

            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid rgba(79, 70, 229, 0.1);
                border-left-color: #4f46e5;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading-text {
                color: #6b7280;
                font-size: 1.1rem;
                margin-bottom: 10px;
            }

            .loading-steps {
                color: #4f46e5;
                font-weight: 500;
                font-size: 0.95rem;
            }

            .results {
                display: none;
                animation: fadeInUp 0.6s ease;
            }

            @keyframes fadeInUp {
                from { 
                    opacity: 0; 
                    transform: translateY(30px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateY(0); 
                }
            }

            .result-section {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05);
                border: 1px solid rgba(229, 231, 235, 0.8);
                transition: all 0.3s ease;
            }

            .result-section:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.1);
            }

            .result-section h3 {
                color: #1f2937;
                margin-bottom: 20px;
                font-size: 1.3rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .section-icon {
                font-size: 1.5rem;
            }

            /* 修改后的文本结果容器样式 */
            .text-result-container {
                position: relative;
            }

            .text-result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }

            .text-result-actions {
                display: flex;
                gap: 8px;
            }

            /* 复制按钮样式 */
            .copy-btn {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 8px 16px;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 0.875rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
            }

            .copy-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
                background: linear-gradient(135deg, #059669 0%, #047857 100%);
            }

            .copy-btn:active {
                transform: translateY(0);
            }

            .copy-btn.copied {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
            }

            .copy-icon {
                font-size: 0.875rem;
            }


            .clear-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(107, 114, 128, 0.4);
                background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
            }

            /* 可编辑文本区域样式 */
            #textResult {
                width: 100%;
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                padding: 20px;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                white-space: pre-wrap;
                overflow-wrap: break-word;
                min-height: 200px;
                max-height: 300px;
                overflow-y: auto;
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                line-height: 1.6;
                font-size: 0.95rem;
                color: #374151;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
                resize: vertical;
                transition: all 0.3s ease;
            }

            #textResult:focus {
                outline: none;
                border-color: #4f46e5;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1), inset 0 2px 4px rgba(0,0,0,0.05);
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            }

            #textResult::placeholder {
                color: #9ca3af;
                font-style: italic;
            }

            .processing-info {
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
                border: 1px solid #a7f3d0;
                color: #065f46;
                font-size: 0.95rem;
            }

            .processing-info strong {
                color: #047857;
            }

            /* 优化后的可视化图像样式 */
            .visualization-container {
                position: relative;
                border-radius: 12px;
                overflow: hidden;
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
            }

            #visualizationImage {
                width: 100%;
                max-width: 100%;
                height: auto;
                max-height: 600px;
                object-fit: contain;
                display: block;
                border-radius: 12px;
                transition: all 0.3s ease;
            }

            #visualizationImage:hover {
                transform: scale(1.02);
                cursor: zoom-in;
            }

            /* 图像放大模态框 */
            .image-modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
                backdrop-filter: blur(5px);
                animation: fadeIn 0.3s ease;
            }

            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                max-width: 95%;
                max-height: 95%;
                border-radius: 12px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.5);
            }

            .modal-image {
                width: 100%;
                height: auto;
                border-radius: 12px;
            }

            .close-modal {
                position: absolute;
                top: 20px;
                right: 30px;
                color: white;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
                z-index: 1001;
                background: rgba(0,0,0,0.5);
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }

            .close-modal:hover {
                background: rgba(0,0,0,0.8);
                transform: scale(1.1);
            }

            .error {
                color: #dc2626;
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                padding: 20px;
                margin: 20px 0;
                border-radius: 12px;
                border: 1px solid #fecaca;
                border-left: 4px solid #dc2626;
                font-weight: 500;
            }

            .warning {
                background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                color: #92400e;
                padding: 15px;
                border-radius: 12px;
                border: 1px solid #fcd34d;
                border-left: 4px solid #f59e0b;
                margin: 15px 0;
                font-weight: 500;
            }

            /* Toast 通知样式 */
            .toast {
                position: fixed;
                top: 85px;
                right: 20px;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
                z-index: 1000;
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 500;
                transform: translateX(100%);
                transition: transform 0.3s ease;
            }

            .toast.show {
                transform: translateX(0);
            }

            .toast.error {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
            }

            /* 响应式设计 */
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                    border-radius: 16px;
                }

                .header h1 {
                    font-size: 2rem;
                }

                .content {
                    padding: 20px;
                }

                .form-section {
                    padding: 20px;
                }

                .checkbox-group {
                    grid-template-columns: 1fr;
                    gap: 15px;
                }

                #visualizationImage {
                    max-height: 400px;
                }

                .text-result-header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 10px;
                }

                .text-result-actions {
                    width: 100%;
                    justify-content: flex-end;
                }

                .toast {
                    right: 10px;
                    left: 10px;
                    transform: translateY(-100%);
                }

                .toast.show {
                    transform: translateY(0);
                }
            }

            /* 滚动条美化 */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }

            /* 动画效果 */
            .fade-in {
                animation: fadeIn 0.5s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            .slide-up {
                animation: slideUp 0.5s ease;
            }

            @keyframes slideUp {
                from { 
                    opacity: 0; 
                    transform: translateY(20px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateY(0); 
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>扭曲矫正 DEMO，欢迎试用！</h1>
            </div>

            <div class="content">
                <div class="form-section">
                    <div class="form-group">
                        <label for="imageFile">📁 选择需要识别的图片</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="imageFile" class="file-input" accept=".png,.jpg,.jpeg,.bmp">
                            <label for="imageFile" class="file-input-label" id="fileLabel">
                                <span class="upload-icon">📤</span>
                                <span id="fileLabelText">点击选择图片文件或拖拽到此处</span>
                            </label>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>⚙️ 处理选项</label>
                        <div class="checkbox-group">
                            <div class="checkbox-item" id="enhancementItem">
                                <input type="checkbox" id="useEnhancement" name="useEnhancement">
                                <label for="useEnhancement">✨ 启用外观增强 (去阴影/去噪)</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="useDewarp" name="useDewarp">
                                <label for="useDewarp">📐 启用扭曲矫正</label>
                            </div>
                        </div>
                        <div id="enhancementWarning" class="warning" style="display: none;">
                            ⚠️ 外观增强模型未加载，此功能不可用
                        </div>
                    </div>

                    <button id="processBtn" class="process-btn">🚀 开始处理</button>
                </div>

                <div id="loading">
                    <div class="spinner"></div>
                    <p class="loading-text">正在处理图像，请稍候...</p>
                    <p id="loadingSteps" class="loading-steps">图像处理步骤将根据您的选择执行</p>
                </div>

                <div id="error" class="error" style="display: none;"></div>

                <div id="results" class="results">
                    <div class="result-section">
                        <h3><span class="section-icon">📝</span>识别结果</h3>
                        <div class="text-result-container">
                            <div class="text-result-header">
                                <span style="color: #6b7280; font-size: 0.875rem;">您可以直接编辑下方的文本内容</span>
                                <div class="text-result-actions">
                                    <button id="copyBtn" class="copy-btn">
                                        <span class="copy-icon">📋</span>
                                        <span id="copyBtnText">复制文本</span>
                                    </button>
                                </div>
                            </div>
                            <textarea id="textResult" placeholder="识别结果将显示在此处，您可以直接编辑..."></textarea>
                        </div>
                        <div id="processingInfo" class="processing-info"></div>
                    </div>

                    <div class="result-section">
                        <h3><span class="section-icon">🎨</span>OCR可视化结果</h3>
                        <div class="visualization-container">
                            <img id="visualizationImage" alt="OCR可视化结果" style="display: none;">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 图像放大模态框 -->
        <div id="imageModal" class="image-modal">
            <span class="close-modal">&times;</span>
            <div class="modal-content">
                <img id="modalImage" class="modal-image" alt="放大的可视化结果">
            </div>
        </div>

        <script>
            // API基础URL - 动态获取当前页面的主机地址
            const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;

            let enhancementAvailable = false;

            document.addEventListener('DOMContentLoaded', function() {
                const imageFileInput = document.getElementById('imageFile');
                const fileLabel = document.getElementById('fileLabel');
                const fileLabelText = document.getElementById('fileLabelText');
                const processBtn = document.getElementById('processBtn');
                const loadingDiv = document.getElementById('loading');
                const loadingSteps = document.getElementById('loadingSteps');
                const resultsDiv = document.getElementById('results');
                const textResultTextarea = document.getElementById('textResult');
                const processingInfoDiv = document.getElementById('processingInfo');
                const errorDiv = document.getElementById('error');

                // 处理选项
                const useEnhancementCheckbox = document.getElementById('useEnhancement');
                const useDewarpCheckbox = document.getElementById('useDewarp');
                const enhancementWarning = document.getElementById('enhancementWarning');
                const enhancementItem = document.getElementById('enhancementItem');

                // 可视化图像和模态框
                const visualizationImage = document.getElementById('visualizationImage');
                const imageModal = document.getElementById('imageModal');
                const modalImage = document.getElementById('modalImage');
                const closeModal = document.querySelector('.close-modal');

                // 文本操作按钮
                const copyBtn = document.getElementById('copyBtn');
                const copyBtnText = document.getElementById('copyBtnText');
                const clearBtn = document.getElementById('clearBtn');

                // 复制功能
                copyBtn.addEventListener('click', function() {
                    const text = textResultTextarea.value;
                    if (!text.trim()) {
                        showToast('没有可复制的内容', 'error');
                        return;
                    }

                    // 使用现代的剪贴板API
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(text).then(function() {
                            showCopySuccess();
                        }).catch(function() {
                            // 如果现代API失败，使用传统方法
                            fallbackCopyTextToClipboard(text);
                        });
                    } else {
                        // 使用传统的复制方法
                        fallbackCopyTextToClipboard(text);
                    }
                });


                // 传统的复制方法（兼容旧浏览器）
                function fallbackCopyTextToClipboard(text) {
                    textResultTextarea.select();
                    textResultTextarea.setSelectionRange(0, 99999); // 移动端兼容

                    try {
                        const successful = document.execCommand('copy');
                        if (successful) {
                            showCopySuccess();
                        } else {
                            showToast('复制失败，请手动选择文本复制', 'error');
                        }
                    } catch (err) {
                        showToast('复制失败，请手动选择文本复制', 'error');
                    }

                    // 取消选择
                    if (window.getSelection) {
                        window.getSelection().removeAllRanges();
                    }
                }

                // 显示复制成功状态
                function showCopySuccess() {
                    // 更改按钮状态
                    copyBtn.classList.add('copied');
                    copyBtnText.textContent = '已复制';
                    
                    // 显示成功提示
                    showToast('文本已复制到剪贴板 ✨');

                    // 2秒后恢复按钮状态
                    setTimeout(function() {
                        copyBtn.classList.remove('copied');
                        copyBtnText.textContent = '复制文本';
                    }, 2000);
                }

                // 显示Toast通知
                function showToast(message, type = 'success') {
                    // 移除已存在的toast
                    const existingToast = document.querySelector('.toast');
                    if (existingToast) {
                        existingToast.remove();
                    }

                    // 创建新的toast
                    const toast = document.createElement('div');
                    toast.className = `toast ${type}`;
                    toast.innerHTML = `
                        <span>${type === 'success' ? '✅' : '❌'}</span>
                        <span>${message}</span>
                    `;

                    document.body.appendChild(toast);

                    // 显示动画
                    setTimeout(() => toast.classList.add('show'), 100);

                    // 3秒后自动消失
                    setTimeout(() => {
                        toast.classList.remove('show');
                        setTimeout(() => toast.remove(), 300);
                    }, 3000);
                }

                // 文件拖拽功能
                fileLabel.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    fileLabel.style.borderColor = '#4f46e5';
                    fileLabel.style.background = 'linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)';
                });

                fileLabel.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    fileLabel.style.borderColor = '#d1d5db';
                    fileLabel.style.background = 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)';
                });

                fileLabel.addEventListener('drop', function(e) {
                    e.preventDefault();
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        imageFileInput.files = files;
                        updateFileLabel(files[0]);
                    }
                    fileLabel.style.borderColor = '#d1d5db';
                    fileLabel.style.background = 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)';
                });

                // 文件选择更新
                imageFileInput.addEventListener('change', function(e) {
                    if (e.target.files.length > 0) {
                        updateFileLabel(e.target.files[0]);
                    }
                    errorDiv.style.display = 'none';
                });

                function updateFileLabel(file) {
                    fileLabelText.textContent = `已选择: ${file.name}`;
                    fileLabel.classList.add('has-file');
                }

                // 图像点击放大功能
                visualizationImage.addEventListener('click', function() {
                    modalImage.src = visualizationImage.src;
                    imageModal.style.display = 'block';
                    document.body.style.overflow = 'hidden';
                });

                // 关闭模态框
                closeModal.addEventListener('click', function() {
                    imageModal.style.display = 'none';
                    document.body.style.overflow = 'auto';
                });

                // 点击模态框背景关闭
                imageModal.addEventListener('click', function(e) {
                    if (e.target === imageModal) {
                        imageModal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                });

                // ESC键关闭模态框
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape' && imageModal.style.display === 'block') {
                        imageModal.style.display = 'none';
                        document.body.style.overflow = 'auto';
                    }
                });

                // 直接检查增强功能可用性（无需健康检查）
                // 假设增强功能默认可用，如果处理时不可用会返回错误
                enhancementAvailable = true;

                // 处理选项变化时更新加载提示
                function updateLoadingSteps() {
                    const steps = [];
                    if (useEnhancementCheckbox.checked && enhancementAvailable) {
                        steps.push('外观增强');
                    }
                    if (useDewarpCheckbox.checked) {
                        steps.push('扭曲矫正');
                    }
                    steps.push('OCR 识别');

                    loadingSteps.textContent = '图像处理步骤：' + steps.join(' → ');
                }

                useEnhancementCheckbox.addEventListener('change', updateLoadingSteps);
                useDewarpCheckbox.addEventListener('change', updateLoadingSteps);
                updateLoadingSteps();

                processBtn.addEventListener('click', function() {
                    const file = imageFileInput.files[0];
                    if (!file) {
                        showError('请先选择图片');
                        return;
                    }

                    // 检查文件格式
                    const validExtensions = ['.png', '.jpg', '.jpeg', '.bmp'];
                    const fileName = file.name.toLowerCase();
                    const isValidFile = validExtensions.some(ext => fileName.endsWith(ext));

                    if (!isValidFile) {
                        showError(`不支持的文件格式。支持的格式: ${validExtensions.join(', ')}`);
                        return;
                    }

                    // 隐藏之前的结果和错误
                    resultsDiv.style.display = 'none';
                    errorDiv.style.display = 'none';

                    // 显示加载指示器
                    loadingDiv.style.display = 'block';
                    loadingDiv.classList.add('fade-in');
                    updateLoadingSteps();

                    // 禁用按钮
                    processBtn.disabled = true;

                    // 创建FormData对象
                    const formData = new FormData();
                    formData.append('image', file);
                    formData.append('use_enhancement', useEnhancementCheckbox.checked);
                    formData.append('use_dewarp', useDewarpCheckbox.checked);

                    // 发送请求
                    fetch(`${API_BASE_URL}/process`, {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('服务器响应错误: ' + response.status);
                        }
                        return response.json();
                    })
                    .then(data => {
                        // 隐藏加载指示器
                        loadingDiv.style.display = 'none';

                        // 启用按钮
                        processBtn.disabled = false;

                        if (data.error) {
                            showError(data.error);
                            return;
                        }

                        // 显示排序后的文本结果到可编辑文本区域
                        textResultTextarea.value = data.sorted_text || '未识别到文本';

                        // 显示处理信息
                        if (data.processing_info) {
                            const info = data.processing_info;
                            processingInfoDiv.innerHTML = `
                                <strong>处理信息:</strong><br>
                                外观增强: ${info.use_enhancement ? '✅ 已启用' : '⏭️ 已跳过'}<br>
                                扭曲矫正: ${info.use_dewarp ? '✅ 已启用' : '⏭️ 已跳过'}<br>
                                检测到文本区域: ${info.text_regions_count} 个
                            `;
                        }

                        // 显示可视化图像
                        if (data.visualization) {
                            visualizationImage.src = 'data:image/png;base64,' + data.visualization;
                            visualizationImage.style.display = 'block';
                            visualizationImage.classList.add('fade-in');
                        }

                        // 显示结果区域
                        resultsDiv.style.display = 'block';
                        resultsDiv.classList.add('slide-up');

                        // 平滑滚动到结果
                        setTimeout(() => {
                            resultsDiv.scrollIntoView({ behavior: 'smooth' });
                        }, 100);

                        // 显示成功提示
                        if (data.sorted_text && data.sorted_text.trim()) {
                            showToast('OCR识别完成！');
                        }
                    })
                    .catch(error => {
                        // 隐藏加载指示器
                        loadingDiv.style.display = 'none';

                        // 启用按钮
                        processBtn.disabled = false;

                        showError('请求错误: ' + error.message);
                    });
                });

                function showError(message) {
                    errorDiv.textContent = message;
                    errorDiv.style.display = 'block';
                    errorDiv.classList.add('fade-in');
                    errorDiv.scrollIntoView({ behavior: 'smooth' });
                }
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # 初始化模型
    print("开始初始化模型...")
    models, device = init_models("cuda:0" if torch.cuda.is_available() else "cpu")

    if models is None or device is None:
        print("模型初始化失败，程序退出")
        sys.exit(1)

    print(f"模型初始化成功，使用设备: {device}")
    print("启动服务器...")
    app.run(host='0.0.0.0', port=5000, debug=False)
