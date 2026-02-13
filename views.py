import os
import json
import pandas as pd
import numpy as np
import torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from model_sakt import SAKT
from model_tsakt import TSAKT
from name_mappings import load_mappings, get_user_name, get_skill_name, get_item_name
from recommendation import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender
from learning_path import AdaptiveLearningPath, LearningPathOptimizer
from chart_config import ChartConfig
from services.PredictionService import PredictionService

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(dataset):
    # 同时检查两种数据文件格式
    data_file = None
    if os.path.exists(os.path.join('data', dataset, 'preprocessed_data.csv')):
        data_file = os.path.join('data', dataset, 'preprocessed_data.csv')
    elif os.path.exists(os.path.join('data', dataset, 'preprocessed_train_data.csv')):
        data_file = os.path.join('data', dataset, 'preprocessed_train_data.csv')
    
    if not data_file:
        return None, None, None, None, None, False
    
    df = pd.read_csv(data_file, sep="\t")
    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)
    
    sakt_path = os.path.join('save/sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=10')
    if not os.path.exists(sakt_path):
        sakt_path = os.path.join('save/sakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5')
    
    if os.path.exists(sakt_path):
        loaded_model = torch.load(sakt_path, map_location=device, weights_only=False)
        sakt_model = loaded_model.to(device)
        sakt_model.eval()
    else:
        sakt_model = None
    
    tsakt_path = os.path.join('save/tsakt', f'{dataset},batch_size=128,max_length=200,encode_pos=False,max_pos=5,tensor_rank=3')
    
    if os.path.exists(tsakt_path):
        loaded_model = torch.load(tsakt_path, map_location=device, weights_only=False)
        tsakt_model = loaded_model.to(device)
        tsakt_model.eval()
        tsakt_available = True
    else:
        tsakt_model = None
        tsakt_available = False
    
    return sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available

def get_available_datasets():
    datasets = []
    # 硬编码的数据集列表，确保即使data目录不存在也能显示选项
    default_datasets = ['assistments09', 'assistments12', 'assistments15', 'algebra05', 'assistments17', 'ednet', 'synthetic']
    
    # 检查data目录是否存在
    if os.path.exists('data'):
        for folder in os.listdir('data'):
            if os.path.isdir(os.path.join('data', folder)):
                if os.path.exists(os.path.join('data', folder, 'preprocessed_train_data.csv')) or os.path.exists(os.path.join('data', folder, 'preprocessed_data.csv')):
                    datasets.append(folder)
    
    # 如果没有找到数据集，使用默认列表
    if not datasets:
        datasets = default_datasets
    
    return sorted(datasets)

def analyze_student_performance(df, user_id):
    user_data = df[df['user_id'] == user_id].sort_values('item_id')
    
    if len(user_data) == 0:
        return None
    
    skill_stats = user_data.groupby('skill_id').agg({
        'correct': ['mean', 'count']
    }).reset_index()
    skill_stats.columns = ['skill_id', 'accuracy', 'count']
    
    skill_stats['mastery'] = skill_stats['accuracy'] * (1 - 1/(1 + skill_stats['count']/10))
    
    return skill_stats

def recommend_questions(skill_stats, num_questions=5, difficulty='balanced'):
    if skill_stats is None or len(skill_stats) == 0:
        return []
    
    if difficulty == 'easy':
        recommended = skill_stats.nlargest(num_questions, 'accuracy')
    elif difficulty == 'hard':
        recommended = skill_stats.nsmallest(num_questions, 'accuracy')
    else:
        recommended = skill_stats.nsmallest(num_questions, 'mastery')
    
    return recommended['skill_id'].tolist()

def prepare_single_sequence(item_ids, skill_ids, labels):
    item_inputs = torch.cat((torch.zeros(1, dtype=torch.long), item_ids + 1))[:-1]
    skill_inputs = torch.cat((torch.zeros(1, dtype=torch.long), skill_ids + 1))[:-1]
    label_inputs = torch.cat((torch.zeros(1, dtype=torch.long), labels))[:-1]
    
    return item_inputs.unsqueeze(0), skill_inputs.unsqueeze(0), label_inputs.unsqueeze(0), \
           item_ids.unsqueeze(0), skill_ids.unsqueeze(0), labels.unsqueeze(0)

def predict_next_question(model, item_ids, skill_ids, labels):
    item_inputs, skill_inputs, label_inputs, item_ids_batch, skill_ids_batch, labels_batch = \
        prepare_single_sequence(item_ids, skill_ids, labels)
    
    item_inputs = item_inputs.to(device)
    skill_inputs = skill_inputs.to(device)
    label_inputs = label_inputs.to(device)
    item_ids_batch = item_ids_batch.to(device)
    skill_ids_batch = skill_ids_batch.to(device)
    
    with torch.no_grad():
        preds = model(item_inputs, skill_inputs, label_inputs, item_ids_batch, skill_ids_batch)
        preds = torch.sigmoid(preds).cpu().numpy()
    
    return preds[0, -1].item()

@csrf_exempt
@require_http_methods(["GET"])
def api_datasets(request):
    datasets = get_available_datasets()
    return JsonResponse({'datasets': datasets})

@csrf_exempt
@require_http_methods(["GET"])
def api_dataset_info(request, dataset):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    users = df['user_id'].unique().tolist()
    user_options = [{'id': uid, 'name': get_user_name(mappings, uid)} for uid in users]
    
    return JsonResponse({
        'num_items': num_items,
        'num_skills': num_skills,
        'num_users': len(users),
        'tsakt_available': tsakt_available,
        'users': user_options
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_user_info(request, dataset, user_id):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    user_data = df[df['user_id'] == int(user_id)]
    
    if len(user_data) == 0:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    return JsonResponse({
        'user_id': int(user_id),
        'user_name': get_user_name(mappings, int(user_id)),
        'total_questions': len(user_data),
        'accuracy': float(user_data['correct'].mean()),
        'unique_skills': int(user_data['skill_id'].nunique())
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_recommendations(request, dataset, user_id):
    num_questions = int(request.GET.get('num', 5))
    difficulty = request.GET.get('difficulty', 'balanced')
    method = request.GET.get('method', 'traditional')
    
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    user_id = int(user_id)
    user_data = df[df['user_id'] == user_id]
    
    if len(user_data) == 0:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    recommendations = []
    
    if method != 'traditional':
        try:
            if method == 'hybrid':
                recommender = HybridRecommender()
            elif method == 'collaborative':
                recommender = CollaborativeFiltering()
            else:
                recommender = ContentBasedRecommender()
            
            recommender.fit(df)
            recommended_items = recommender.recommend_for_user(user_id, num_questions)
            
            skill_stats = analyze_student_performance(df, user_id)
            
            for item_id in recommended_items:
                item_data = df[df['item_id'] == item_id].iloc[0]
                skill_id = item_data['skill_id']
                skill_name = get_skill_name(mappings, skill_id)
                
                mastery = 0
                accuracy = 0
                if skill_stats is not None and skill_id in skill_stats['skill_id'].values:
                    skill_data = skill_stats[skill_stats['skill_id'] == skill_id].iloc[0]
                    mastery = float(skill_data['mastery'])
                    accuracy = float(skill_data['accuracy'])
                
                recommendations.append({
                    'item_id': int(item_id),
                    'skill_id': int(skill_id),
                    'skill_name': skill_name,
                    'mastery': mastery,
                    'accuracy': accuracy
                })
        except Exception as e:
            pass
    
    if len(recommendations) == 0:
        skill_stats = analyze_student_performance(df, user_id)
        recommended_skills = recommend_questions(skill_stats, num_questions, difficulty)
        
        for skill_id in recommended_skills:
            skill_data = skill_stats[skill_stats['skill_id'] == skill_id].iloc[0]
            skill_name = get_skill_name(mappings, skill_id)
            
            recommendations.append({
                'skill_id': int(skill_id),
                'skill_name': skill_name,
                'mastery': float(skill_data['mastery']),
                'accuracy': float(skill_data['accuracy'])
            })
    
    return JsonResponse({'recommendations': recommendations})

@csrf_exempt
@require_http_methods(["GET"])
def api_prediction(request, dataset, user_id):
    model_choice = request.GET.get('model', 'sakt')
    
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    model = sakt_model if model_choice == 'sakt' else tsakt_model
    
    if model is None:
        return JsonResponse({'error': 'Model not available'}, status=400)
    
    user_data = df[df['user_id'] == int(user_id)]
    
    if len(user_data) == 0:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    recent_answers = user_data.tail(10)
    item_ids = torch.tensor(recent_answers['item_id'].values, dtype=torch.long)
    skill_ids = torch.tensor(recent_answers['skill_id'].values, dtype=torch.long)
    labels = torch.tensor(recent_answers['correct'].values, dtype=torch.long)
    
    prediction = predict_next_question(model, item_ids, skill_ids, labels)
    
    return JsonResponse({'prediction': prediction})

@csrf_exempt
@require_http_methods(["GET"])
def api_skill_stats(request, dataset, user_id):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    user_id = int(user_id)
    skill_stats = analyze_student_performance(df, user_id)
    
    if skill_stats is None:
        return JsonResponse({'error': 'No data found'}, status=404)
    
    skill_stats['skill_name'] = skill_stats['skill_id'].apply(lambda x: get_skill_name(mappings, x))
    
    skills = []
    for _, row in skill_stats.iterrows():
        skills.append({
            'skill_id': int(row['skill_id']),
            'skill_name': row['skill_name'],
            'accuracy': float(row['accuracy']),
            'mastery': float(row['mastery']),
            'count': int(row['count'])
        })
    
    return JsonResponse({'skills': skills})

@csrf_exempt
@require_http_methods(["GET"])
def api_learning_trend(request, dataset, user_id):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    user_data = df[df['user_id'] == int(user_id)].sort_values('item_id')
    
    if len(user_data) == 0:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    user_data = user_data.reset_index(drop=True)
    user_data['cumulative_accuracy'] = user_data['correct'].expanding().mean()
    
    trend = []
    for i, row in user_data.iterrows():
        trend.append({
            'index': int(i),
            'cumulative_accuracy': float(row['cumulative_accuracy'])
        })
    
    return JsonResponse({'trend': trend})

@csrf_exempt
@require_http_methods(["GET"])
def api_error_analysis(request, dataset, user_id):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    user_data = df[df['user_id'] == int(user_id)]
    
    if len(user_data) == 0:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    wrong_answers = user_data[user_data['correct'] == 0]
    
    if len(wrong_answers) == 0:
        return JsonResponse({'errors': []})
    
    skill_error_counts = wrong_answers['skill_id'].value_counts().head(10)
    
    errors = []
    for skill_id, count in skill_error_counts.items():
        errors.append({
            'skill_id': int(skill_id),
            'skill_name': get_skill_name(mappings, int(skill_id)),
            'count': int(count)
        })
    
    return JsonResponse({'errors': errors})

@csrf_exempt
@require_http_methods(["GET"])
def api_learning_path(request, dataset, user_id):
    learning_goal = request.GET.get('goal', 'comprehensive')
    max_length = int(request.GET.get('max_length', 10))
    
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        return JsonResponse({'error': '数据集不存在'}, status=404)
    
    mappings = load_mappings()
    
    try:
        learning_path_optimizer = LearningPathOptimizer(df)
        learning_path = learning_path_optimizer.adaptive_path.recommend_learning_path(int(user_id), max_length=max_length)
        
        if learning_path:
            path = []
            for i, skill_id in enumerate(learning_path, 1):
                skill_name = get_skill_name(mappings, skill_id)
                path.append({
                    'step': i,
                    'skill_id': int(skill_id),
                    'skill_name': skill_name
                })
            
            return JsonResponse({'path': path})
        else:
            return JsonResponse({'path': []})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def recommendation_page(request):
    datasets = get_available_datasets()
    return render(request, 'recommendation.html', {'datasets': datasets})

def analysis_page(request):
    datasets = get_available_datasets()
    return render(request, 'analysis.html', {'datasets': datasets})

def learning_path_page(request):
    datasets = get_available_datasets()
    return render(request, 'learning_path.html', {'datasets': datasets})

def assessment_page(request):
    datasets = get_available_datasets()
    return render(request, 'assessment.html', {'datasets': datasets})

def upload_page(request):
    return render(request, 'upload.html')

@csrf_exempt
@require_http_methods(["POST"])
def api_ocr(request):
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        engine = request.POST.get('engine', 'paddleocr')
        print(f"收到的engine参数: {engine}")
        print(f"收到的image文件: {image_file.name}")
        
        # 设置PaddleOCR相关环境变量
        import os
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        os.environ['PADDLEOCR_OFFLINE'] = 'True'
        os.environ['PADDLEX_HOME'] = os.path.join(os.path.dirname(__file__), 'paddlex_cache')
        os.environ['PADDLE_HOME'] = os.environ['PADDLEX_HOME']
        os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(__file__), 'modelscope_cache')
        
        # 创建缓存目录
        os.makedirs(os.environ['PADDLEX_HOME'], exist_ok=True)
        os.makedirs(os.environ['MODELSCOPE_CACHE'], exist_ok=True)
        
        # 保存上传的图片
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_image_path = temp_file.name
        
        try:
            # 导入PaddleOCR
            from paddleocr import PaddleOCR
            import numpy as np
            
            # 初始化OCR引擎
            full_text = ""
            recognized_text = []
            
            if engine == 'paddleocr-vl':
                # 使用PaddleOCR-VL（复杂模型）
                try:
                    # 尝试导入PaddleOCRVL
                    from paddleocr import PaddleOCRVL
                    
                    # 尝试初始化PaddleOCRVL
                    try:
                        ocr = PaddleOCRVL()
                        
                        # 使用PaddleOCR-VL的predict方法
                        result = ocr.predict(temp_image_path)
                        print(f"PaddleOCR-VL预测结果: {result}")
                        print(f"结果类型: {type(result)}")
                        
                        # 提取识别结果 - 使用app.py中的详细解析逻辑
                        recognized_text = []
                        text_lines = []
                        
                        try:
                            if isinstance(result, list):
                                print(f"OCR结果长度: {len(result)}")
                                for i, res in enumerate(result):
                                    print(f"第{i}项类型: {type(res)}")
                                    if hasattr(res, 'rec_texts'):
                                        rec_texts = res.rec_texts
                                        print(f"第{i}项rec_texts类型: {type(rec_texts)}")
                                        if isinstance(rec_texts, list):
                                            text_lines.extend(rec_texts)
                                            for text in rec_texts:
                                                print(f"从rec_texts获取: {text}")
                                        elif isinstance(rec_texts, str):
                                            text_lines.append(rec_texts)
                                            print(f"从rec_texts获取: {rec_texts}")
                                    elif hasattr(res, 'text'):
                                        text_lines.append(res.text)
                                        print(f"从text获取: {res.text}")
                                    elif isinstance(res, str):
                                        text_lines.append(res)
                                        print(f"直接获取字符串: {res}")
                                    elif isinstance(res, dict):
                                        print(f"第{i}项是字典，键: {list(res.keys())}")
                                        if 'text' in res:
                                            text_lines.append(res['text'])
                                            print(f"从字典text获取: {res['text']}")
                                        elif 'rec_texts' in res:
                                            text_lines.extend(res['rec_texts'])
                                            print(f"从字典rec_texts获取: {res['rec_texts']}")
                        except Exception as e:
                            print(f"解析PaddleOCR-VL结果时出错: {str(e)}")
                        
                        full_text = '\n'.join(text_lines)
                        print(f"PaddleOCR-VL分支的full_text: {full_text}")
                    except Exception as vl_error:
                        # 如果PaddleOCRVL初始化或识别失败，回退到标准PaddleOCR
                        print(f"PaddleOCR-VL错误: {str(vl_error)}")
                        # 回退到标准PaddleOCR
                        ocr = PaddleOCR(
                            use_angle_cls=True,
                            lang='ch',
                            det_db_thresh=0.3,
                            det_db_box_thresh=0.5,
                            det_db_unclip_ratio=1.5
                        )
                        
                        # 调整图片大小以提高识别率
                        image = Image.open(temp_image_path)
                        max_size = 1024
                        width, height = image.size
                        if max(width, height) > max_size:
                            ratio = max_size / max(width, height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                        
                        # 保存图片到临时文件以便调试
                        debug_image_path = os.path.join(os.path.dirname(__file__), 'debug_image_fallback.jpg')
                        image.save(debug_image_path)
                        print(f"回退逻辑调试图片已保存到: {debug_image_path}")
                        
                        # 执行OCR识别
                        print(f"开始执行OCR识别，使用标准PaddleOCR（回退逻辑）")
                        print(f"图片大小: {image.size}")
                        print(f"图片模式: {image.mode}")
                        
                        # 转换图片为numpy数组
                        img_array = np.array(image)
                        print(f"图片数组形状: {img_array.shape}")
                        print(f"图片数组数据类型: {img_array.dtype}")
                        
                        # 执行OCR识别
                        result = ocr.ocr(img_array)
                        print(f"OCR识别结果: {result}")
                        print(f"OCR结果类型: {type(result)}")
                        print(f"OCR结果长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                        
                        # 使用详细解析逻辑
                        text_lines = []
                        try:
                            if result:
                                if isinstance(result, list):
                                    print(f"OCR结果是列表，长度: {len(result)}")
                                    for i, item in enumerate(result):
                                        print(f"第{i}项类型: {type(item)}")
                                        print(f"第{i}项内容: {item}")
                                        if isinstance(item, list):
                                            print(f"第{i}项是列表，长度: {len(item)}")
                                            if len(item) > 0:
                                                print(f"第{i}项第一个元素类型: {type(item[0])}")
                                                print(f"第{i}项第一个元素内容: {item[0]}")
                                            # 处理 PP-OCRv5_server 的返回格式
                                            for j, line in enumerate(item):
                                                print(f"第{i}-{j}行类型: {type(line)}")
                                                print(f"第{i}-{j}行内容: {line}")
                                                if isinstance(line, (list, tuple)):
                                                    print(f"第{i}-{j}行长度: {len(line)}")
                                                    # 尝试不同的结果格式
                                                    if len(line) >= 2:
                                                        print(f"第{i}-{j}行第2个元素类型: {type(line[1])}")
                                                        print(f"第{i}-{j}行第2个元素内容: {line[1]}")
                                                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                                                            text_lines.append(line[1][0])
                                                            print(f"识别到文字: {line[1][0]}")
                                                        elif isinstance(line[1], str):
                                                            text_lines.append(line[1])
                                                            print(f"识别到文字: {line[1]}")
                                                    elif len(line) >= 1:
                                                        if isinstance(line[0], str):
                                                            text_lines.append(line[0])
                                                            print(f"识别到文字: {line[0]}")
                                        elif hasattr(item, 'rec_texts'):
                                            rec_texts = item.rec_texts
                                            print(f"第{i}项rec_texts类型: {type(rec_texts)}")
                                            print(f"第{i}项rec_texts内容: {rec_texts}")
                                            if isinstance(rec_texts, list):
                                                text_lines.extend(rec_texts)
                                                for text in rec_texts:
                                                    print(f"识别到文字: {text}")
                                            elif isinstance(rec_texts, str):
                                                text_lines.append(rec_texts)
                                                print(f"识别到文字: {rec_texts}")
                                        elif isinstance(item, dict):
                                            print(f"第{i}项是字典，键: {list(item.keys())}")
                                            if 'text' in item:
                                                text_lines.append(item['text'])
                                                print(f"识别到文字: {item['text']}")
                                            elif 'rec_texts' in item:
                                                text_lines.extend(item['rec_texts'])
                                                print(f"识别到文字: {item['rec_texts']}")
                                        elif hasattr(item, 'text'):
                                            text_lines.append(item.text)
                                            print(f"识别到文字: {item.text}")
                                        else:
                                            print(f"第{i}项内容: {item}")
                        except Exception as e:
                            print(f"解析回退OCR结果时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
                        
                        full_text = '\n'.join(text_lines)
                        print(f"回退逻辑最终识别文本: '{full_text}'")
                        print(f"回退逻辑full_text长度: {len(full_text)}")
                        print(f"回退逻辑text_lines列表: {text_lines}")
                        print(f"回退逻辑text_lines长度: {len(text_lines)}")
                except ImportError:
                    # 如果PaddleOCRVL不可用，回退到标准PaddleOCR
                    print(f"PaddleOCRVL模块导入失败，回退到标准PaddleOCR")
                    ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch',
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                        det_db_unclip_ratio=1.5
                    )
                    
                    # 调整图片大小以提高识别率
                    image = Image.open(temp_image_path)
                    max_size = 1024
                    width, height = image.size
                    if max(width, height) > max_size:
                        ratio = max_size / max(width, height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 保存图片到临时文件以便调试
                    debug_image_path = os.path.join(os.path.dirname(__file__), 'debug_image_import_error.jpg')
                    image.save(debug_image_path)
                    print(f"ImportError回退逻辑调试图片已保存到: {debug_image_path}")
                    
                    # 执行OCR识别
                    print(f"开始执行OCR识别，使用标准PaddleOCR（ImportError回退逻辑）")
                    print(f"图片大小: {image.size}")
                    print(f"图片模式: {image.mode}")
                    
                    # 转换图片为numpy数组
                    img_array = np.array(image)
                    print(f"图片数组形状: {img_array.shape}")
                    print(f"图片数组数据类型: {img_array.dtype}")
                    
                    # 执行OCR识别
                    result = ocr.ocr(img_array)
                    print(f"OCR识别结果: {result}")
                    print(f"OCR结果类型: {type(result)}")
                    print(f"OCR结果长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                    
                    # 使用详细解析逻辑
                    text_lines = []
                    try:
                        if result:
                            if isinstance(result, list):
                                print(f"OCR结果是列表，长度: {len(result)}")
                                for i, item in enumerate(result):
                                    print(f"第{i}项类型: {type(item)}")
                                    print(f"第{i}项内容: {item}")
                                    if isinstance(item, list):
                                        print(f"第{i}项是列表，长度: {len(item)}")
                                        if len(item) > 0:
                                            print(f"第{i}项第一个元素类型: {type(item[0])}")
                                            print(f"第{i}项第一个元素内容: {item[0]}")
                                        # 处理 PP-OCRv5_server 的返回格式
                                        for j, line in enumerate(item):
                                            print(f"第{i}-{j}行类型: {type(line)}")
                                            print(f"第{i}-{j}行内容: {line}")
                                            if isinstance(line, (list, tuple)):
                                                print(f"第{i}-{j}行长度: {len(line)}")
                                                # 尝试不同的结果格式
                                                if len(line) >= 2:
                                                    print(f"第{i}-{j}行第2个元素类型: {type(line[1])}")
                                                    print(f"第{i}-{j}行第2个元素内容: {line[1]}")
                                                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                                                        text_lines.append(line[1][0])
                                                        print(f"识别到文字: {line[1][0]}")
                                                    elif isinstance(line[1], str):
                                                        text_lines.append(line[1])
                                                        print(f"识别到文字: {line[1]}")
                                                elif len(line) >= 1:
                                                    if isinstance(line[0], str):
                                                        text_lines.append(line[0])
                                                        print(f"识别到文字: {line[0]}")
                                    elif hasattr(item, 'rec_texts'):
                                        rec_texts = item.rec_texts
                                        print(f"第{i}项rec_texts类型: {type(rec_texts)}")
                                        print(f"第{i}项rec_texts内容: {rec_texts}")
                                        if isinstance(rec_texts, list):
                                            text_lines.extend(rec_texts)
                                            for text in rec_texts:
                                                print(f"识别到文字: {text}")
                                        elif isinstance(rec_texts, str):
                                            text_lines.append(rec_texts)
                                            print(f"识别到文字: {rec_texts}")
                                    elif isinstance(item, dict):
                                        print(f"第{i}项是字典，键: {list(item.keys())}")
                                        if 'text' in item:
                                            text_lines.append(item['text'])
                                            print(f"识别到文字: {item['text']}")
                                        elif 'rec_texts' in item:
                                            text_lines.extend(item['rec_texts'])
                                            print(f"识别到文字: {item['rec_texts']}")
                                    elif hasattr(item, 'text'):
                                        text_lines.append(item.text)
                                        print(f"识别到文字: {item.text}")
                                    else:
                                        print(f"第{i}项内容: {item}")
                    except Exception as e:
                        print(f"解析ImportError回退OCR结果时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    full_text = '\n'.join(text_lines)
                    print(f"ImportError回退逻辑最终识别文本: '{full_text}'")
                    print(f"ImportError回退逻辑full_text长度: {len(full_text)}")
                    print(f"ImportError回退逻辑text_lines列表: {text_lines}")
                    print(f"ImportError回退逻辑text_lines长度: {len(text_lines)}")
            else:
                # 使用标准PaddleOCR（简单模型）
                ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.5
                )
                
                # 调整图片大小以提高识别率
                image = Image.open(temp_image_path)
                max_size = 1024
                width, height = image.size
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # 执行OCR识别
                print(f"开始执行OCR识别，使用标准PaddleOCR")
                print(f"图片大小: {image.size}")
                print(f"图片模式: {image.mode}")
                
                # 保存图片到临时文件以便调试
                debug_image_path = os.path.join(os.path.dirname(__file__), 'debug_image.jpg')
                image.save(debug_image_path)
                print(f"调试图片已保存到: {debug_image_path}")
                
                # 转换图片为numpy数组
                img_array = np.array(image)
                print(f"图片数组形状: {img_array.shape}")
                print(f"图片数组数据类型: {img_array.dtype}")
                
                # 执行OCR识别
                result = ocr.ocr(img_array)
                print(f"OCR识别结果: {result}")
                print(f"OCR结果类型: {type(result)}")
                print(f"OCR结果长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                
                # 处理识别结果 - 使用app.py中的详细解析逻辑
                recognized_text = []
                text_lines = []
                
                try:
                    if result:
                        if isinstance(result, list):
                            print(f"OCR结果是列表，长度: {len(result)}")
                            for i, item in enumerate(result):
                                print(f"第{i}项类型: {type(item)}")
                                print(f"第{i}项内容: {item}")
                                if isinstance(item, list):
                                    print(f"第{i}项是列表，长度: {len(item)}")
                                    if len(item) > 0:
                                        print(f"第{i}项第一个元素类型: {type(item[0])}")
                                        print(f"第{i}项第一个元素内容: {item[0]}")
                                    # 处理 PP-OCRv5_server 的返回格式
                                    for j, line in enumerate(item):
                                        print(f"第{i}-{j}行类型: {type(line)}")
                                        print(f"第{i}-{j}行内容: {line}")
                                        if isinstance(line, (list, tuple)):
                                            print(f"第{i}-{j}行长度: {len(line)}")
                                            # 尝试不同的结果格式
                                            if len(line) >= 2:
                                                print(f"第{i}-{j}行第2个元素类型: {type(line[1])}")
                                                print(f"第{i}-{j}行第2个元素内容: {line[1]}")
                                                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                                                    text_lines.append(line[1][0])
                                                    print(f"识别到文字: {line[1][0]}")
                                                elif isinstance(line[1], str):
                                                    text_lines.append(line[1])
                                                    print(f"识别到文字: {line[1]}")
                                            elif len(line) >= 1:
                                                if isinstance(line[0], str):
                                                    text_lines.append(line[0])
                                                    print(f"识别到文字: {line[0]}")
                                elif hasattr(item, 'rec_texts'):
                                    rec_texts = item.rec_texts
                                    print(f"第{i}项rec_texts类型: {type(rec_texts)}")
                                    print(f"第{i}项rec_texts内容: {rec_texts}")
                                    if isinstance(rec_texts, list):
                                        text_lines.extend(rec_texts)
                                        for text in rec_texts:
                                            print(f"识别到文字: {text}")
                                    elif isinstance(rec_texts, str):
                                        text_lines.append(rec_texts)
                                        print(f"识别到文字: {rec_texts}")
                                elif isinstance(item, dict):
                                    print(f"第{i}项是字典，键: {list(item.keys())}")
                                    if 'text' in item:
                                        text_lines.append(item['text'])
                                        print(f"识别到文字: {item['text']}")
                                    elif 'rec_texts' in item:
                                        text_lines.extend(item['rec_texts'])
                                        print(f"识别到文字: {item['rec_texts']}")
                                elif hasattr(item, 'text'):
                                    text_lines.append(item.text)
                                    print(f"识别到文字: {item.text}")
                                else:
                                    print(f"第{i}项内容: {item}")
                except Exception as e:
                    print(f"解析OCR结果时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                full_text = '\n'.join(text_lines)
                print(f"最终识别文本: '{full_text}'")
                print(f"full_text长度: {len(full_text)}")
                print(f"text_lines列表: {text_lines}")
                print(f"text_lines长度: {len(text_lines)}")
            
            print(f"返回结果前的full_text: {full_text}")
            print(f"full_text长度: {len(full_text)}")
            
            if not full_text.strip():
                print(f"警告：识别结果为空")
                return JsonResponse({'success': False, 'error': '未能识别出文字，请尝试使用更清晰的图片或手动输入'}, status=400)
            
            return JsonResponse({'success': True, 'text': full_text})
        finally:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def api_save_question(request):
    try:
        print(f"收到保存题目请求")
        
        # 获取上传的图片
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'success': False, 'error': '未上传图片'}, status=400)
        
        # 获取题目信息
        content = request.POST.get('content', '')
        skill_id = request.POST.get('skill_id', '0')
        difficulty = request.POST.get('difficulty', '中等')
        dataset_name = request.POST.get('dataset_name', 'custom_images')
        
        print(f"题目内容: {content}")
        print(f"知识点ID: {skill_id}")
        print(f"难度级别: {difficulty}")
        print(f"数据集名称: {dataset_name}")
        
        if not content.strip():
            return JsonResponse({'success': False, 'error': '题目内容不能为空'}, status=400)
        
        # 创建数据集目录
        os.makedirs('data', exist_ok=True)
        os.makedirs(f'data/{dataset_name}', exist_ok=True)
        
        # 保存图片
        image_path = f'data/{dataset_name}/images'
        os.makedirs(image_path, exist_ok=True)
        
        # 获取当前图片数量
        existing_images = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_filename = f"{len(existing_images) + 1}.jpg"
        image_full_path = os.path.join(image_path, image_filename)
        
        # 保存图片文件
        with open(image_full_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
        
        print(f"图片已保存到: {image_full_path}")
        
        # 保存题目信息
        questions_path = os.path.join(f'data/{dataset_name}', 'questions.json')
        
        if os.path.exists(questions_path):
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        else:
            questions = []
        
        new_question = {
            "id": len(questions) + 1,
            "content": content,
            "skill_id": int(skill_id),
            "difficulty": difficulty,
            "image": image_filename
        }
        
        questions.append(new_question)
        
        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        print(f"题目信息已保存到: {questions_path}")
        print(f"保存的题目: {new_question}")
        
        return JsonResponse({
            'success': True,
            'message': f'题目已保存到: {questions_path}',
            'question': new_question
        })
    except Exception as e:
        print(f"保存题目时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def api_mappings(request, dataset):
    try:
        mappings = load_mappings(dataset)
        return JsonResponse({'success': True, 'mappings': mappings})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def api_ids(request, dataset, name_type):
    try:
        # 检查data目录是否存在
        if not os.path.exists('data'):
            # 如果data目录不存在，返回默认的学生ID列表
            if name_type == 'student':
                return JsonResponse({'success': True, 'ids': list(range(1, 11))})  # 默认10个学生
            elif name_type == 'skill':
                return JsonResponse({'success': True, 'ids': list(range(1, 21))})  # 默认20个知识点
            elif name_type == 'item':
                return JsonResponse({'success': True, 'ids': list(range(1, 51))})  # 默认50个题目
        
        # 同时检查两种数据文件格式
        data_file = None
        if os.path.exists(os.path.join('data', dataset, 'preprocessed_data.csv')):
            data_file = os.path.join('data', dataset, 'preprocessed_data.csv')
        elif os.path.exists(os.path.join('data', dataset, 'preprocessed_train_data.csv')):
            data_file = os.path.join('data', dataset, 'preprocessed_train_data.csv')
        else:
            # 如果数据集文件夹不存在，返回默认ID列表
            if name_type == 'student':
                return JsonResponse({'success': True, 'ids': list(range(1, 11))})  # 默认10个学生
            elif name_type == 'skill':
                return JsonResponse({'success': True, 'ids': list(range(1, 21))})  # 默认20个知识点
            elif name_type == 'item':
                return JsonResponse({'success': True, 'ids': list(range(1, 51))})  # 默认50个题目
        
        df = pd.read_csv(data_file, sep="\t")
        
        ids = []
        if name_type == 'student':
            ids = sorted(df['user_id'].unique().tolist())
        elif name_type == 'skill':
            ids = sorted(df['skill_id'].unique().tolist())
        elif name_type == 'item':
            ids = sorted(df['item_id'].unique().tolist())
        
        return JsonResponse({'success': True, 'ids': ids})
    except Exception as e:
        # 如果发生任何错误，返回默认ID列表
        if name_type == 'student':
            return JsonResponse({'success': True, 'ids': list(range(1, 11))})  # 默认10个学生
        elif name_type == 'skill':
            return JsonResponse({'success': True, 'ids': list(range(1, 21))})  # 默认20个知识点
        elif name_type == 'item':
            return JsonResponse({'success': True, 'ids': list(range(1, 51))})  # 默认50个题目

@csrf_exempt
@require_http_methods(["POST"])
def api_save_name(request):
    try:
        data = json.loads(request.body)
        dataset = data.get('dataset')
        name_type = data.get('type')
        id = data.get('id')
        name = data.get('name')
        
        if not dataset or not name_type or id is None or not name:
            return JsonResponse({'success': False, 'error': '缺少必要参数'}, status=400)
        
        mappings = load_mappings(dataset)
        
        if name_type == 'student':
            mappings['user_names'][id] = name
        elif name_type == 'skill':
            mappings['skill_names'][id] = name
        elif name_type == 'item':
            mappings['item_names'][id] = name
        
        # 保存映射
        mappings_path = os.path.join('data', dataset, 'name_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        return JsonResponse({'success': True, 'message': '名称已保存'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def api_auto_generate_name(request):
    try:
        data = json.loads(request.body)
        dataset = data.get('dataset')
        name_type = data.get('type')
        id = data.get('id')
        
        if not dataset or not name_type or id is None:
            return JsonResponse({'success': False, 'error': '缺少必要参数'}, status=400)
        
        mappings = load_mappings(dataset)
        
        if name_type == 'student':
            mappings['user_names'][id] = f"学生{id}"
        elif name_type == 'skill':
            mappings['skill_names'][id] = f"知识点{id}"
        elif name_type == 'item':
            mappings['item_names'][id] = f"题目{id}"
        
        # 保存映射
        mappings_path = os.path.join('data', dataset, 'name_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        return JsonResponse({'success': True, 'message': '已生成默认名称'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def api_batch_generate_names(request):
    try:
        data = json.loads(request.body)
        dataset = data.get('dataset')
        name_type = data.get('type')
        
        if not dataset or not name_type:
            return JsonResponse({'success': False, 'error': '缺少必要参数'}, status=400)
        
        if not os.path.exists(os.path.join('data', dataset, 'preprocessed_data.csv')):
            return JsonResponse({'success': False, 'error': '数据集不存在'}, status=404)
        
        df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
        mappings = load_mappings(dataset)
        
        if name_type == 'skill':
            unique_skills = sorted(df['skill_id'].unique())
            for skill_id in unique_skills:
                mappings['skill_names'][skill_id] = f"知识点{skill_id}"
        elif name_type == 'student':
            unique_users = sorted(df['user_id'].unique())
            for user_id in unique_users:
                mappings['user_names'][user_id] = f"学生{user_id}"
        elif name_type == 'item':
            unique_items = sorted(df['item_id'].unique())
            for item_id in unique_items:
                mappings['item_names'][item_id] = f"题目{item_id}"
        
        # 保存映射
        mappings_path = os.path.join('data', dataset, 'name_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        return JsonResponse({'success': True, 'message': '已批量生成名称'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def api_heatmap(request, dataset, user_id):
    sakt_model, tsakt_model, df, num_items, num_skills, tsakt_available = load_models(dataset)
    
    if df is None:
        # 如果数据集不存在，生成模拟的热力图数据
        import random
        heatmap_data = []
        for i in range(1, 21):  # 模拟20个知识点
            heatmap_data.append({
                'skill_id': i,
                'mastery': round(random.uniform(0.3, 0.95), 2),  # 随机掌握度
                'count': random.randint(5, 20)  # 随机题目数量
            })
        return JsonResponse({
            'data': heatmap_data,
            'total_skills': len(heatmap_data),
            'total_questions': sum(item['count'] for item in heatmap_data)
        })
    
    user_data = df[df['user_id'] == int(user_id)]
    
    if len(user_data) == 0:
        # 如果用户不存在，生成模拟的热力图数据
        import random
        heatmap_data = []
        for i in range(1, 21):  # 模拟20个知识点
            heatmap_data.append({
                'skill_id': i,
                'mastery': round(random.uniform(0.3, 0.95), 2),  # 随机掌握度
                'count': random.randint(5, 20)  # 随机题目数量
            })
        return JsonResponse({
            'data': heatmap_data,
            'total_skills': len(heatmap_data),
            'total_questions': sum(item['count'] for item in heatmap_data)
        })
    
    skill_stats = user_data.groupby('skill_id').agg({
        'correct': ['mean', 'count']
    }).reset_index()
    
    skill_stats.columns = ['skill_id', 'mastery', 'count']
    
    heatmap_data = []
    for _, row in skill_stats.iterrows():
        heatmap_data.append({
            'skill_id': int(row['skill_id']),
            'mastery': float(row['mastery']),
            'count': int(row['count'])
        })
    
    return JsonResponse({
        'data': heatmap_data,
        'total_skills': len(heatmap_data),
        'total_questions': len(user_data)
    })

@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """预测学生学习表现"""
    try:
        data = json.loads(request.body)
        student_id = data.get('student_id')
        skill_id = data.get('skill_id')
        dataset = data.get('dataset', 'assistments_2009_2010')

        if not student_id or not skill_id:
            return JsonResponse({'code': 400, 'msg': '缺少必要参数'}, status=400)

        if not os.path.exists(os.path.join('data', dataset, 'preprocessed_data.csv')):
            return JsonResponse({'code': 404, 'msg': '数据集不存在'}, status=404)

        df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
        
        # 获取学生数据
        student_data = df[df['user_id'] == int(student_id)].to_dict('records')
        
        # 创建预测服务
        prediction_service = PredictionService()
        
        # 进行预测
        results = prediction_service.predict(student_data, student_id, skill_id)
        
        return JsonResponse({
            'code': 200,
            'msg': '预测成功',
            'data': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'code': 500, 'msg': f'预测失败: {str(e)}'}, status=500)

def prediction_page(request):
    """预测页面"""
    return render(request, 'prediction.html')

def model_comparison_page(request):
    """模型对比页面"""
    return render(request, 'model_comparison.html')

def login_page(request):
    """登录页面"""
    return render(request, 'login.html')

def register_page(request):
    """注册页面"""
    return render(request, 'register.html')

def heatmap_page(request):
    """热力图页面"""
    datasets = get_available_datasets()
    return render(request, 'heatmap.html', {'datasets': datasets})
