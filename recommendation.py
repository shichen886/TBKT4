import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class CollaborativeFiltering:
    def __init__(self, min_interactions=5):
        self.min_interactions = min_interactions
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        
    def fit(self, data):
        """
        训练协同过滤模型
        
        Args:
            data: DataFrame with columns [user_id, item_id, correct]
        """
        # 限制用户和物品数量，防止内存溢出（增加到500个）
        top_users = data['user_id'].value_counts().head(500).index
        data = data[data['user_id'].isin(top_users)]
        
        top_items = data['item_id'].value_counts().head(500).index
        data = data[data['item_id'].isin(top_items)]
        
        if len(data) < 50:
            self.user_item_matrix = None
            self.item_user_matrix = None
            self.user_similarity = None
            self.item_similarity = None
            return
        
        # 创建用户-物品矩阵
        self.user_item_matrix = data.pivot_table(
            index='user_id',
            columns='item_id',
            values='correct',
            fill_value=0
        )
        
        # 创建物品-用户矩阵
        self.item_user_matrix = data.pivot_table(
            index='item_id',
            columns='user_id',
            values='correct',
            fill_value=0
        )
        
        # 计算用户相似度（基于用户的行为）
        # 使用稀疏矩阵或限制计算范围
        try:
            # 只计算前200个用户的相似度，减少内存使用
            if len(self.user_item_matrix) > 200:
                self.user_similarity = None
            else:
                self.user_similarity = cosine_similarity(self.user_item_matrix)
                self.user_similarity = pd.DataFrame(
                    self.user_similarity,
                    index=self.user_item_matrix.index,
                    columns=self.user_item_matrix.index
                )
        except Exception as e:
            self.user_similarity = None
        
        # 计算物品相似度（基于物品被用户回答的情况）
        try:
            # 只计算前200个物品的相似度，减少内存使用
            if len(self.item_user_matrix) > 200:
                self.item_similarity = None
            else:
                self.item_similarity = cosine_similarity(self.item_user_matrix)
                self.item_similarity = pd.DataFrame(
                    self.item_similarity,
                    index=self.item_user_matrix.index,
                    columns=self.item_user_matrix.index
                )
        except Exception as e:
            self.item_similarity = None
        
    def recommend_for_user(self, user_id, top_k=10, method='user_based'):
        """
        为用户推荐题目
        
        Args:
            user_id: 用户ID
            top_k: 推荐的题目数量
            method: 'user_based' 或 'item_based'
        
        Returns:
            推荐的题目列表
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        if method == 'user_based':
            return self._user_based_recommend(user_id, top_k)
        else:
            return self._item_based_recommend(user_id, top_k)
    
    def _user_based_recommend(self, user_id, top_k):
        """基于用户的协同过滤"""
        # 获取相似用户
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)
        
        # 获取用户已回答的题目
        user_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        # 推荐分数
        recommendations = defaultdict(float)
        
        for similar_user, similarity in similar_users.items():
            if similar_user == user_id or similarity < 0.1:
                continue
            
            # 获取相似用户回答过但当前用户未回答的题目
            similar_user_items = self.user_item_matrix.loc[similar_user]
            for item_id, score in similar_user_items.items():
                if item_id not in user_items and score > 0:
                    recommendations[item_id] += similarity * score
        
        # 排序并返回top_k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:top_k]]
    
    def _item_based_recommend(self, user_id, top_k):
        """基于物品的协同过滤"""
        # 获取用户已回答的题目
        user_items = self.user_item_matrix.loc[user_id]
        answered_items = user_items[user_items > 0].index.tolist()
        
        if not answered_items:
            return []
        
        # 推荐分数
        recommendations = defaultdict(float)
        
        # 对于用户已回答的每个题目，找到相似的题目
        for item_id in answered_items:
            similar_items = self.item_similarity[item_id].sort_values(ascending=False)
            
            for similar_item, similarity in similar_items.items():
                if similar_item == item_id or similar_item in answered_items or similarity < 0.1:
                    continue
                
                recommendations[similar_item] += similarity * user_items[item_id]
        
        # 排序并返回top_k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:top_k]]
    
    def get_similar_items(self, item_id, top_k=10):
        """
        获取相似的题目
        
        Args:
            item_id: 题目ID
            top_k: 返回的相似题目数量
        
        Returns:
            相似题目列表及其相似度
        """
        if item_id not in self.item_similarity.index:
            return []
        
        similar_items = self.item_similarity[item_id].sort_values(ascending=False)
        similar_items = similar_items[similar_items.index != item_id]
        
        return similar_items.head(top_k).items()


class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.feature_similarity = None
        
    def fit(self, data):
        """
        训练基于内容的推荐模型
        
        Args:
            data: DataFrame with columns [item_id, skill_id, difficulty, ...]
        """
        # 限制数据量，减少内存使用
        top_items = data['item_id'].value_counts().head(1000).index
        data = data[data['item_id'].isin(top_items)]
        
        if len(data) < 50:
            self.item_features = None
            self.feature_similarity = None
            return
        
        # 提取题目特征
        self.item_features = data[['item_id', 'skill_id']].drop_duplicates()
        
        # 创建难度映射
        if 'difficulty' in data.columns:
            difficulty_map = {'简单': 1, '中等': 2, '困难': 3}
            self.item_features['difficulty'] = data.groupby('item_id')['difficulty'].first().reindex(self.item_features['item_id'])
            self.item_features['difficulty_encoded'] = self.item_features['difficulty'].map(difficulty_map).fillna(2)
        else:
            # 基于正确率计算难度
            difficulty_by_item = data.groupby('item_id')['correct'].agg(lambda x: 3 - (x.mean() * 2)).round().clip(1, 3)
            self.item_features['difficulty_encoded'] = difficulty_by_item.reindex(self.item_features['item_id']).fillna(2)
        
        # 创建特征矩阵
        feature_matrix = self.item_features[['skill_id', 'difficulty_encoded']].values
        
        # 计算特征相似度
        try:
            self.feature_similarity = cosine_similarity(feature_matrix)
            self.feature_similarity = pd.DataFrame(
                self.feature_similarity,
                index=self.item_features['item_id'],
                columns=self.item_features['item_id']
            )
        except Exception as e:
            self.feature_similarity = None
        
    def recommend_for_user(self, user_data, user_id, top_k=10):
        """
        基于用户历史行为推荐题目
        
        Args:
            user_data: 用户的历史答题数据
            user_id: 用户ID
            top_k: 推荐的题目数量
        
        Returns:
            推荐的题目列表
        """
        # 获取用户的历史答题
        user_history = user_data[user_data['user_id'] == user_id]
        
        if user_history.empty:
            return []
        
        # 获取用户最常回答的知识点
        skill_counts = user_history['skill_id'].value_counts()
        top_skills = skill_counts.head(3).index.tolist()
        
        # 获取用户平均难度
        avg_difficulty = user_history['difficulty_encoded'].mean()
        
        # 推荐分数
        recommendations = defaultdict(float)
        
        # 对于用户历史中的每个题目，找到相似的题目
        for _, row in user_history.iterrows():
            item_id = row['item_id']
            if item_id not in self.feature_similarity.index:
                continue
            
            similar_items = self.feature_similarity[item_id].sort_values(ascending=False)
            
            for similar_item, similarity in similar_items.items():
                if similar_item == item_id:
                    continue
                
                # 检查是否是用户擅长的知识点
                item_features = self.item_features[self.item_features['item_id'] == similar_item]
                if item_features['skill_id'].values[0] in top_skills:
                    recommendations[similar_item] += similarity * 1.5
                else:
                    recommendations[similar_item] += similarity
        
        # 排序并返回top_k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:top_k]]
    
    def get_similar_items(self, item_id, top_k=10):
        """
        获取相似的题目（基于特征）
        
        Args:
            item_id: 题目ID
            top_k: 返回的相似题目数量
        
        Returns:
            相似题目列表及其相似度
        """
        if item_id not in self.feature_similarity.index:
            return []
        
        similar_items = self.feature_similarity[item_id].sort_values(ascending=False)
        similar_items = similar_items[similar_items.index != item_id]
        
        return similar_items.head(top_k).items()


class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_recommender = CollaborativeFiltering()
        self.cb_recommender = ContentBasedRecommender()
        
    def fit(self, data):
        """
        训练混合推荐模型
        
        Args:
            data: DataFrame with columns [user_id, item_id, skill_id, difficulty, correct]
        """
        try:
            self.cf_recommender.fit(data)
        except Exception as e:
            pass
        
        try:
            self.cb_recommender.fit(data)
        except Exception as e:
            pass
        
    def recommend_for_user(self, user_id, top_k=10):
        """
        混合推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐的题目数量
        
        Returns:
            推荐的题目列表
        """
        # 推荐分数
        recommendations = defaultdict(float)
        
        # 获取协同过滤推荐
        try:
            cf_recs = self.cf_recommender.recommend_for_user(user_id, top_k * 2)
            for i, item_id in enumerate(cf_recs):
                recommendations[item_id] += self.cf_weight * (len(cf_recs) - i)
        except Exception as e:
            pass
        
        # 获取基于内容的推荐
        try:
            cb_recs = self.cb_recommender.recommend_for_user(
                self.cf_recommender.user_item_matrix.reset_index().melt(
                    id_vars='user_id',
                    var_name='item_id',
                    value_name='correct'
                ).query('correct > 0'),
                user_id,
                top_k * 2
            )
            for i, item_id in enumerate(cb_recs):
                recommendations[item_id] += self.cb_weight * (len(cb_recs) - i)
        except Exception as e:
            pass
        
        # 排序并返回top_k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:top_k]]