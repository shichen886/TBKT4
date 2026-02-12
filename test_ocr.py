# 测试本地OCR模型
import os
from paddleocr import PaddleOCR

print("正在测试本地OCR模型...")

# 获取当前目录
app_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(app_dir, 'models')

print(f"模型目录: {model_dir}")

# 检查模型文件
print("\n检查模型文件:")

# 检测模型
det_model_path = os.path.join(model_dir, 'ch_PP-OCRv4_det_infer')
if os.path.exists(os.path.join(det_model_path, 'ch_PP-OCRv4_det_infer')):
    det_model_path = os.path.join(det_model_path, 'ch_PP-OCRv4_det_infer')
det_model_exists = os.path.exists(os.path.join(det_model_path, 'inference.pdmodel'))
print(f"检测模型: {det_model_path} {'✅' if det_model_exists else '❌'}")

# 识别模型
rec_model_path = os.path.join(model_dir, 'ch_PP-OCRv4_rec_infer')
if os.path.exists(os.path.join(rec_model_path, 'ch_PP-OCRv4_rec_infer')):
    rec_model_path = os.path.join(rec_model_path, 'ch_PP-OCRv4_rec_infer')
rec_model_exists = os.path.exists(os.path.join(rec_model_path, 'inference.pdmodel'))
print(f"识别模型: {rec_model_path} {'✅' if rec_model_exists else '❌'}")

# 尝试初始化OCR
if det_model_exists and rec_model_exists:
    print("\n尝试初始化OCR引擎...")
    try:
        # 设置环境变量
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        os.environ['PADDLEOCR_OFFLINE'] = 'True'
        
        # 使用新的参数名称
        ocr = PaddleOCR(
            text_detection_model_dir=det_model_path,
            text_recognition_model_dir=rec_model_path,
            lang='ch',
            use_textline_orientation=False
        )
        print("✅ OCR引擎初始化成功！")
        print("模型加载成功，OCR功能可用。")
        
        # 测试OCR功能
        print("\n测试OCR识别功能...")
        # 创建一个简单的测试图像
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 创建测试图像
        img = Image.new('RGB', (400, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        # 尝试使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        d.text((10, 30), "测试题目：1+1=?", fill=(0, 0, 0), font=font)
        
        # 保存测试图像
        test_image_path = os.path.join(app_dir, 'test_ocr_image.png')
        img.save(test_image_path)
        print(f"创建测试图像: {test_image_path}")
        
        # 测试识别
        result = ocr.ocr(np.array(img))
        if result and result[0]:
            print("✅ 识别成功！")
            for line in result[0]:
                print(f"识别结果: {line[1][0]}")
        else:
            print("❌ 识别失败")
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n可能的原因:")
        print("1. PaddleOCR版本不兼容")
        print("2. 模型文件版本不匹配")
        print("3. 环境变量问题")
else:
    print("\n❌ 模型文件不存在，无法初始化OCR引擎。")