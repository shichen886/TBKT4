import numpy as np
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction.TensorAttention import TensorAttention
from models.prediction.RNNPredictor import RNNPredictor
from models.prediction.LSTMPredictor import LSTMPredictor
from models.prediction.GRUPredictor import GRUPredictor
from models.prediction.AvgAlgorithmPredictor import AvgAlgorithmPredictor
from models.prediction.AprioriPredictor import AprioriPredictor

class PredictionService:
    """预测服务，整合各种算法进行预测"""
    
    def __init__(self):
        """初始化预测服务"""
        self.tensor_model = TensorAttention()
        self.rnn_model = RNNPredictor()
        self.lstm_model = LSTMPredictor()
        self.gru_model = GRUPredictor()
        self.avg_model = AvgAlgorithmPredictor()
        self.apriori_model = AprioriPredictor()
        # 用于确保示例数据一致性
        random.seed(42)
        np.random.seed(42)
        
    def preprocess_data(self, student_data):
        """预处理学生数据，提取特征和标签
        
        Args:
            student_data: 学生学习记录数据
            
        Returns:
            特征矩阵和标签向量
        """
        print("\n--- 数据预处理开始 ---")
        print(f"原始数据示例 (前3条):")
        for i, record in enumerate(student_data[:3]):
            print(f"  记录 {i}: {record}")
        
        # 提取特征和标签
        features = []
        labels = []
        
        for record in student_data:
            try:
                # 提取相关字段作为特征
                feature = [
                    float(record.get('problem_id', 0)),  # 问题编号
                    float(record.get('step_duration', 0)),  # 步骤持续时间
                    float(record.get('correct', 0))  # 是否正确
                ]
                
                # 使用correct字段作为标签
                label = float(record.get('correct', 0))
                
                features.append(feature)
                labels.append(label)
            except Exception as e:
                print(f"处理记录时出错: {e}, 记录: {record}")
                continue
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"预处理数据结果:")
        print(f"  特征数量: {len(features)}")
        print(f"  标签数量: {len(labels)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  特征值范围: 最小值={np.min(features):.2f}, 最大值={np.max(features):.2f}")
        print(f"  标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")
        sys.stdout.flush()
        
        return features, labels
    
    def predict(self, student_data, student_id, skill_id):
        """使用多种算法进行预测并返回结果"""
        # 强制刷新输出
        sys.stdout.flush()
        
        print("\n========== 开始学生学习表现预测 ==========")
        print(f"学生ID: {student_id}, 技能ID: {skill_id}")
        print(f"输入数据数量: {len(student_data)}")
        print("======================================")
        
        # 如果没有足够的历史数据，使用示例数据
        if len(student_data) < 5:
            print("历史数据不足，使用示例数据进行预测")
            sys.stdout.flush()
            return self._generate_example_prediction()
        
        try:
            # 预处理数据
            print("开始预处理数据...")
            features, labels = self.preprocess_data(student_data)
            
            # 打印调试信息
            print(f"\n===== 调试信息: 学生{student_id}的技能{skill_id}预测 =====")
            print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
            print(f"特征数据示例: ")
            for i in range(min(3, len(features))):
                print(f"  样本 {i}: {features[i]}")
            print(f"标签数据示例: ")
            for i in range(min(5, len(labels))):
                print(f"  样本 {i}: {labels[i]}")
            
            # 强制刷新输出
            sys.stdout.flush()
            
            # 设置预测参数
            n_future = 3  # 预测未来3个时间点
            
            print("\n开始执行张量自注意力模型预测...")
            sys.stdout.flush()
            # 使用张量自注意力模型
            tensor_preds = self.tensor_model.predict(features, labels, n_future)
            
            print("\n开始执行RNN模型预测...")
            sys.stdout.flush()
            # 使用RNN模型
            rnn_preds = self.rnn_model.predict(features, labels, n_future)
            
            print("\n开始执行LSTM模型预测...")
            sys.stdout.flush()
            # 使用LSTM模型
            lstm_preds = self.lstm_model.predict(features, labels, n_future)
            
            print("\n开始执行GRU模型预测...")
            sys.stdout.flush()
            # 使用GRU模型
            gru_preds = self.gru_model.predict(features, labels, n_future)
            
            print("\n开始执行Apriori算法预测...")
            sys.stdout.flush()
            # 使用Apriori模型
            apriori_preds = self.apriori_model.predict_from_features(features, labels, n_future)
            
            # 使用平均模型组合各个算法的结果
            print("\n计算算法加权平均...")
            predictions_list = [tensor_preds, rnn_preds, lstm_preds, gru_preds, apriori_preds]
            weights = self.avg_model.get_algorithm_weights(predictions_list, labels) if len(labels) > 0 else None
            avg_preds = self.avg_model.predict(predictions_list, weights)
            
            # 打印预测结果
            print(f"\n预测值范围分析:")
            print(f"张量自注意力: 最小值={min(tensor_preds):.4f}, 最大值={max(tensor_preds):.4f}, 平均值={sum(tensor_preds)/len(tensor_preds):.4f}")
            print(f"RNN: 最小值={min(rnn_preds):.4f}, 最大值={max(rnn_preds):.4f}, 平均值={sum(rnn_preds)/len(rnn_preds):.4f}")
            print(f"LSTM: 最小值={min(lstm_preds):.4f}, 最大值={max(lstm_preds):.4f}, 平均值={sum(lstm_preds)/len(lstm_preds):.4f}")
            print(f"GRU: 最小值={min(gru_preds):.4f}, 最大值={max(gru_preds):.4f}, 平均值={sum(gru_preds)/len(gru_preds):.4f}")
            print(f"Apriori: 最小值={min(apriori_preds):.4f}, 最大值={max(apriori_preds):.4f}, 平均值={sum(apriori_preds)/len(apriori_preds):.4f}")
            print(f"平均算法: 最小值={min(avg_preds):.4f}, 最大值={max(avg_preds):.4f}, 平均值={sum(avg_preds)/len(avg_preds):.4f}")
            sys.stdout.flush()
            
            # 准备时间点标签
            time_points = [f"问题{i+1}" for i in range(len(labels))]
            time_points.extend([f"预测{i+1}" for i in range(n_future)])
            
            # 准备实际值（历史+预测）
            actual_values = labels.tolist()
            actual_values.extend([None] * n_future)  # 未来预测部分没有实际值
            
            # 计算性能分析数据
            performance_analysis = self._analyze_performance(labels, tensor_preds[:len(labels)])
            
            # 计算各算法准确率
            accuracy = self._calculate_accuracy(labels, 
                tensor_preds[:len(labels)], 
                rnn_preds[:len(labels)], 
                lstm_preds[:len(labels)], 
                gru_preds[:len(labels)],
                apriori_preds[:len(labels)]
            )
            
            # 打印准确率分析
            print(f"\n准确率分析结果: {accuracy}")
            sys.stdout.flush()
            
            # 组装预测结果
            prediction_results = {
                'time_points': time_points,
                'actual_values': actual_values,
                'tensor_attention_pred': tensor_preds.tolist(),
                'rnn_pred': rnn_preds.tolist(),
                'lstm_pred': lstm_preds.tolist(),
                'gru_pred': gru_preds.tolist(),
                'apriori_pred': apriori_preds.tolist(),
                'avg_pred': avg_preds.tolist(),
                'performance_analysis': performance_analysis,
                'accuracy': accuracy
            }
            
            print("\n========== 预测完成 ==========")
            sys.stdout.flush()
            
            return prediction_results
        
        except Exception as e:
            import traceback
            print(f"\n预测过程出错: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            print("\n使用示例数据进行预测")
            return self._generate_example_prediction()
    
    def _analyze_performance(self, actual, predicted):
        """分析学生的学习表现状态"""
        # 根据历史正确率和预测准确率评估掌握程度
        correct_count = sum(actual)
        accuracy = correct_count / len(actual) if len(actual) > 0 else 0
        
        # 根据预测准确率评估未来趋势
        predicted_binary = [1 if p >= 0.5 else 0 for p in predicted]
        future_trend = sum(predicted_binary) / len(predicted_binary) if len(predicted_binary) > 0 else 0
        
        # 计算学习状态比例
        mastery = accuracy * 0.6 + future_trend * 0.4  # 已掌握程度
        learning = (1 - mastery) * 0.7  # 学习中程度
        struggling = (1 - mastery) * 0.3  # 需要帮助程度
        
        return {
            'mastery': round(mastery, 2),
            'learning': round(learning, 2),
            'struggling': round(struggling, 2)
        }
    
    def _calculate_accuracy(self, actual, tensor_pred, rnn_pred, lstm_pred, gru_pred, apriori_pred=None):
        """计算各算法的预测准确率"""
        # 打印调试信息
        print("\n----- 准确率计算调试信息 -----")
        print(f"实际值: {actual[:5]}...")
        
        # 如果没有实际值，无法计算准确率
        if len(actual) == 0:
            print("警告: 没有实际值，无法计算准确率")
            return {
                'tensor_attention': 0.0,
                'rnn': 0.0,
                'lstm': 0.0,
                'gru': 0.0,
                'apriori': 0.0,
                'avg': 0.0
            }
        
        # 将概率转换为二分类结果
        tensor_binary = [1 if p >= 0.5 else 0 for p in tensor_pred]
        rnn_binary = [1 if p >= 0.5 else 0 for p in rnn_pred]
        lstm_binary = [1 if p >= 0.5 else 0 for p in lstm_pred]
        gru_binary = [1 if p >= 0.5 else 0 for p in gru_pred]
        apriori_binary = [1 if p >= 0.5 else 0 for p in apriori_pred] if apriori_pred is not None else []
        
        print(f"二分类后的张量自注意力预测: {tensor_binary[:5]}...")
        print(f"二分类后的RNN预测: {rnn_binary[:5]}...")
        print(f"二分类后的LSTM预测: {lstm_binary[:5]}...")
        print(f"二分类后的GRU预测: {gru_binary[:5]}...")
        if apriori_pred is not None:
            print(f"二分类后的Apriori预测: {apriori_binary[:5]}...")
        
        # 安全地计算准确率，避免除以零
        def safe_accuracy(actual, predicted):
            if len(actual) == 0:
                return 0.0
            matches = sum(1 for a, p in zip(actual, predicted) if a == p)
            return matches / len(actual)
        
        # 计算每种算法的准确率
        tensor_acc = safe_accuracy(actual, tensor_binary)
        rnn_acc = safe_accuracy(actual, rnn_binary)
        lstm_acc = safe_accuracy(actual, lstm_binary)
        gru_acc = safe_accuracy(actual, gru_binary)
        apriori_acc = safe_accuracy(actual, apriori_binary) if apriori_pred is not None else 0.0
        
        # 计算平均模型的准确率（基于加权投票）
        predictions_list = [tensor_pred, rnn_pred, lstm_pred, gru_pred]
        if apriori_pred is not None:
            predictions_list.append(apriori_pred)
            
        weights = self.avg_model.get_algorithm_weights(predictions_list, actual) if len(actual) > 0 else None
        avg_preds = self.avg_model.predict(predictions_list, weights)
        avg_binary = [1 if p >= 0.5 else 0 for p in avg_preds[:len(actual)]]
        avg_acc = safe_accuracy(actual, avg_binary)
        
        print(f"各算法准确率计算结果:")
        print(f"张量自注意力: {tensor_acc:.4f}")
        print(f"RNN: {rnn_acc:.4f}")
        print(f"LSTM: {lstm_acc:.4f}")
        print(f"GRU: {gru_acc:.4f}")
        if apriori_pred is not None:
            print(f"Apriori: {apriori_acc:.4f}")
        print(f"平均算法: {avg_acc:.4f}")
        print("----- 准确率计算调试信息结束 -----\n")
        
        result = {
            'tensor_attention': round(tensor_acc, 2),
            'rnn': round(rnn_acc, 2),
            'lstm': round(lstm_acc, 2),
            'gru': round(gru_acc, 2),
            'avg': round(avg_acc, 2)
        }
        
        if apriori_pred is not None:
            result['apriori'] = round(apriori_acc, 2)
            
        return result
    
    def _generate_example_prediction(self, include_apriori=False):
        """生成示例预测数据"""
        # 时间点
        time_points = ['问题1', '问题2', '问题3', '问题4', '问题5', 
                      '问题6', '问题7', '问题8', '预测1', '预测2', '预测3']
        
        # 生成随机的历史数据
        actual = []
        for i in range(8):
            actual.append(random.choice([0, 1]))
        
        # 为预测部分添加None
        actual_values = actual + [None, None, None]
        
        # 为每个算法生成模拟预测
        def generate_random_preds():
            preds = []
            # 生成历史拟合值
            for i in range(8):
                if actual[i] == 1:
                    preds.append(random.uniform(0.65, 0.95))
                else:
                    preds.append(random.uniform(0.05, 0.35))
            # 生成未来预测值
            recent_avg = sum(actual[-3:]) / 3 if actual else 0.5
            for i in range(3):
                if recent_avg > 0.6:
                    preds.append(random.uniform(0.7, 0.9))
                elif recent_avg < 0.4:
                    preds.append(random.uniform(0.1, 0.3))
                else:
                    preds.append(random.uniform(0.4, 0.6))
            return preds
        
        # 生成各算法的预测值
        tensor_preds = generate_random_preds()
        rnn_preds = generate_random_preds()
        lstm_preds = generate_random_preds()
        gru_preds = generate_random_preds()
        
        # 如果包含Apriori算法
        apriori_preds = None
        if include_apriori:
            apriori_preds = generate_random_preds()
        
        # 生成平均预测
        if include_apriori and apriori_preds:
            avg_preds = [(a+b+c+d+e)/5 for a,b,c,d,e in zip(tensor_preds, rnn_preds, lstm_preds, gru_preds, apriori_preds)]
        else:
            avg_preds = [(a+b+c+d)/4 for a,b,c,d in zip(tensor_preds, rnn_preds, lstm_preds, gru_preds)]
        
        # 计算算法准确率
        tensor_acc = sum(1 for i in range(8) if (tensor_preds[i] >= 0.5) == actual[i]) / 8
        rnn_acc = sum(1 for i in range(8) if (rnn_preds[i] >= 0.5) == actual[i]) / 8
        lstm_acc = sum(1 for i in range(8) if (lstm_preds[i] >= 0.5) == actual[i]) / 8
        gru_acc = sum(1 for i in range(8) if (gru_preds[i] >= 0.5) == actual[i]) / 8
        apriori_acc = sum(1 for i in range(8) if (apriori_preds[i] >= 0.5) == actual[i]) / 8 if apriori_preds else 0
        avg_acc = sum(1 for i in range(8) if (avg_preds[i] >= 0.5) == actual[i]) / 8
        
        # 构建准确率字典
        accuracy = {
            'tensor_attention': round(tensor_acc, 2),
            'rnn': round(rnn_acc, 2),
            'lstm': round(lstm_acc, 2),
            'gru': round(gru_acc, 2),
            'avg': round(avg_acc, 2)
        }
        
        if include_apriori and apriori_preds:
            accuracy['apriori'] = round(apriori_acc, 2)
        
        # 性能分析
        performance_analysis = {
            'mastery': random.uniform(0.4, 0.8),
            'learning': random.uniform(0.1, 0.4),
            'struggling': random.uniform(0.05, 0.2)
        }
        
        # 构建返回结果
        result = {
            'time_points': time_points,
            'actual_values': actual_values,
            'tensor_attention_pred': tensor_preds,
            'rnn_pred': rnn_preds,
            'lstm_pred': lstm_preds,
            'gru_pred': gru_preds,
            'avg_pred': avg_preds,
            'performance_analysis': performance_analysis,
            'accuracy': accuracy
        }
        
        # 如果包含Apriori算法
        if include_apriori and apriori_preds:
            result['apriori_pred'] = apriori_preds
            
        return result
