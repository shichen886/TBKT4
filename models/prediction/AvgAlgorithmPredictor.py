import numpy as np
import random

class AvgAlgorithmPredictor:
    """算法平均预测器，结合多个算法结果进行预测"""
    
    def __init__(self):
        """初始化平均预测器"""
        # 用于确保示例数据一致性
        random.seed(42)
        np.random.seed(42)
    
    def predict(self, predictions_list, weights=None):
        """使用多个算法的预测结果进行加权平均
        
        Args:
            predictions_list: 各个算法的预测结果列表，每个元素是一个预测数组
            weights: 各个算法的权重，如果为None则使用相等权重
            
        Returns:
            加权平均的预测结果
        """
        if not predictions_list:
            return np.array([])
        
        # 确保所有预测结果长度相同
        prediction_length = len(predictions_list[0])
        for preds in predictions_list:
            if len(preds) != prediction_length:
                raise ValueError("所有预测结果的长度必须相同")
        
        # 如果没有提供权重，则使用相等权重
        n_algorithms = len(predictions_list)
        if weights is None:
            weights = np.ones(n_algorithms) / n_algorithms
        else:
            # 确保权重和为1
            weights = np.array(weights) / sum(weights)
        
        # 计算加权平均
        avg_predictions = np.zeros(prediction_length)
        for i, preds in enumerate(predictions_list):
            avg_predictions += weights[i] * np.array(preds)
        
        return avg_predictions
    
    def analyze_predictions(self, predictions_list, true_labels=None):
        """分析各个算法的预测结果
        
        Args:
            predictions_list: 各个算法的预测结果列表
            true_labels: 真实标签，如果提供则计算各算法的准确率
            
        Returns:
            各算法的分析结果
        """
        if not predictions_list:
            return {}
        
        n_algorithms = len(predictions_list)
        analysis = {
            "算法数量": n_algorithms,
            "预测长度": len(predictions_list[0]),
            "各算法均值": [np.mean(preds) for preds in predictions_list],
            "各算法标准差": [np.std(preds) for preds in predictions_list],
            "各算法最大值": [np.max(preds) for preds in predictions_list],
            "各算法最小值": [np.min(preds) for preds in predictions_list]
        }
        
        # 如果提供了真实标签，计算各算法的准确率
        if true_labels is not None:
            true_labels = np.array(true_labels)
            accuracies = []
            for preds in predictions_list:
                # 将连续值转换为二分类预测
                binary_preds = (np.array(preds[:len(true_labels)]) > 0.5).astype(int)
                accuracy = np.mean(binary_preds == true_labels)
                accuracies.append(accuracy)
            analysis["各算法准确率"] = accuracies
        
        return analysis
    
    def get_algorithm_weights(self, predictions_list, true_labels):
        """根据各算法在历史数据上的表现确定最优权重
        
        Args:
            predictions_list: 各个算法的预测结果列表
            true_labels: 真实标签
            
        Returns:
            各算法的最优权重
        """
        if not predictions_list or true_labels is None:
            return None
        
        n_algorithms = len(predictions_list)
        if n_algorithms == 1:
            return [1.0]
        
        # 简化实现：根据准确率确定权重
        accuracies = []
        for preds in predictions_list:
            # 只使用有真实标签的部分进行评估
            pred_values = np.array(preds[:len(true_labels)])
            binary_preds = (pred_values > 0.5).astype(int)
            accuracy = np.mean(binary_preds == true_labels)
            accuracies.append(max(0.01, accuracy))  # 避免零权重
        
        # 将准确率归一化为权重
        weights = np.array(accuracies) / sum(accuracies)
        
        return weights.tolist()
    
    def get_confidence_intervals(self, predictions_list, confidence=0.95):
        """计算预测结果的置信区间
        
        Args:
            predictions_list: 各个算法的预测结果列表
            confidence: 置信度，默认为0.95
            
        Returns:
            每个时间点的预测均值和置信区间
        """
        if not predictions_list:
            return None
        
        prediction_length = len(predictions_list[0])
        means = np.zeros(prediction_length)
        lower_bounds = np.zeros(prediction_length)
        upper_bounds = np.zeros(prediction_length)
        
        # 对每个时间点计算均值和置信区间
        for t in range(prediction_length):
            # 收集所有算法在该时间点的预测
            point_predictions = [preds[t] for preds in predictions_list]
            
            # 计算均值
            mean = np.mean(point_predictions)
            means[t] = mean
            
            # 计算标准差
            std = np.std(point_predictions)
            
            # 计算置信区间
            # 简化实现：使用正态分布的置信区间
            z = 1.96  # 对应95%置信度的z值
            if confidence != 0.95:
                # 其他置信度可以添加对应的z值
                if confidence == 0.90:
                    z = 1.645
                elif confidence == 0.99:
                    z = 2.576
            
            margin = z * std / np.sqrt(len(predictions_list))
            lower_bounds[t] = max(0, mean - margin)
            upper_bounds[t] = min(1, mean + margin)
        
        return {
            "means": means,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        } 
