document.addEventListener('DOMContentLoaded', function() {
    const datasetSelect = document.getElementById('datasetSelect');
    const userSelect = document.getElementById('userSelect');
    const difficultySelect = document.getElementById('difficultySelect');
    const numQuestions = document.getElementById('numQuestions');
    const numQuestionsValue = document.getElementById('numQuestionsValue');
    const methodSelect = document.getElementById('methodSelect');
    const generateBtn = document.getElementById('generateBtn');
    
    const userInfoCard = document.getElementById('userInfoCard');
    const recommendationsCard = document.getElementById('recommendationsCard');
    const predictionCard = document.getElementById('predictionCard');
    
    numQuestions.addEventListener('input', function() {
        numQuestionsValue.textContent = this.value;
    });
    
    loadDatasets();
    
    datasetSelect.addEventListener('change', function() {
        if (this.value) {
            loadUsers(this.value);
        } else {
            userSelect.innerHTML = '<option value="">请先选择数据集</option>';
            userSelect.disabled = true;
        }
    });
    
    generateBtn.addEventListener('click', generateRecommendations);
    
    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets/');
            const data = await response.json();
            
            datasetSelect.innerHTML = '<option value="">选择数据集</option>';
            data.datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset;
                option.textContent = dataset;
                datasetSelect.appendChild(option);
            });
        } catch (error) {
            console.error('加载数据集失败:', error);
        }
    }
    
    async function loadUsers(dataset) {
        try {
            const response = await fetch(`/api/dataset/${dataset}/`);
            const data = await response.json();
            
            userSelect.innerHTML = '<option value="">选择学生</option>';
            data.users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.id;
                option.textContent = user.name;
                userSelect.appendChild(option);
            });
            userSelect.disabled = false;
        } catch (error) {
            console.error('加载用户失败:', error);
        }
    }
    
    async function generateRecommendations() {
        const dataset = datasetSelect.value;
        const userId = userSelect.value;
        const difficulty = difficultySelect.value;
        const num = numQuestions.value;
        const method = methodSelect.value;
        
        if (!dataset || !userId) {
            alert('请选择数据集和学生');
            return;
        }
        
        generateBtn.disabled = true;
        generateBtn.textContent = '生成中...';
        
        try {
            await Promise.all([
                loadUserInfo(dataset, userId),
                loadRecommendations(dataset, userId, difficulty, num, method),
                loadPrediction(dataset, userId)
            ]);
        } catch (error) {
            console.error('生成推荐失败:', error);
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = '生成推荐';
        }
    }
    
    async function loadUserInfo(dataset, userId) {
        try {
            const response = await fetch(`/api/user/${dataset}/${userId}/`);
            const data = await response.json();
            
            document.getElementById('totalQuestions').textContent = data.total_questions;
            document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('uniqueSkills').textContent = data.unique_skills;
            
            userInfoCard.style.display = 'block';
        } catch (error) {
            console.error('加载用户信息失败:', error);
        }
    }
    
    async function loadRecommendations(dataset, userId, difficulty, num, method) {
        try {
            const response = await fetch(`/api/recommend/${dataset}/${userId}/?difficulty=${difficulty}&num=${num}&method=${method}`);
            const data = await response.json();
            
            const list = document.getElementById('recommendationsList');
            list.innerHTML = '';
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach((item, index) => {
                    const div = document.createElement('div');
                    div.className = 'recommendation-item';
                    
                    const masteryPercent = (item.mastery * 100).toFixed(2);
                    const accuracyPercent = (item.accuracy * 100).toFixed(2);
                    
                    let status = '';
                    let statusClass = '';
                    if (item.accuracy < 0.5) {
                        status = '需加强';
                        statusClass = 'error';
                    } else if (item.accuracy < 0.7) {
                        status = '一般';
                        statusClass = 'warning';
                    } else {
                        status = '良好';
                        statusClass = 'success';
                    }
                    
                    div.innerHTML = `
                        <div class="rec-number">${index + 1}</div>
                        <div class="rec-content">
                            <div class="rec-title">${item.skill_name}</div>
                            <div class="rec-progress">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${masteryPercent}%"></div>
                                </div>
                                <span class="progress-text">掌握度: ${masteryPercent}% | 正确率: ${accuracyPercent}%</span>
                            </div>
                        </div>
                        <div class="rec-status ${statusClass}">${status}</div>
                    `;
                    
                    list.appendChild(div);
                });
            } else {
                list.innerHTML = '<p class="no-data">暂无推荐题目</p>';
            }
            
            recommendationsCard.style.display = 'block';
        } catch (error) {
            console.error('加载推荐失败:', error);
        }
    }
    
    async function loadPrediction(dataset, userId) {
        try {
            const response = await fetch(`/api/predict/${dataset}/${userId}/`);
            const data = await response.json();
            
            const predictionPercent = (data.prediction * 100).toFixed(2);
            document.getElementById('predictionValue').textContent = predictionPercent + '%';
            
            let status = '';
            if (data.prediction > 0.7) {
                status = '准备充分';
                document.getElementById('predictionStatus').className = 'prediction-status success';
            } else if (data.prediction > 0.5) {
                status = '需要复习';
                document.getElementById('predictionStatus').className = 'prediction-status warning';
            } else {
                status = '建议先学习';
                document.getElementById('predictionStatus').className = 'prediction-status error';
            }
            document.getElementById('predictionStatus').textContent = status;
            
            predictionCard.style.display = 'block';
        } catch (error) {
            console.error('加载预测失败:', error);
        }
    }
});