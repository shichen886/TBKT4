document.addEventListener('DOMContentLoaded', function() {
    const datasetSelect = document.getElementById('datasetSelect');
    const userSelect = document.getElementById('userSelect');
    const goalSelect = document.getElementById('goalSelect');
    const pathLength = document.getElementById('pathLength');
    const pathLengthValue = document.getElementById('pathLengthValue');
    const generatePathBtn = document.getElementById('generatePathBtn');
    
    const pathCard = document.getElementById('pathCard');
    const pathContainer = document.getElementById('pathContainer');
    
    // 检查元素是否存在
    if (pathLength && pathLengthValue) {
        pathLength.addEventListener('input', function() {
            pathLengthValue.textContent = this.value;
        });
    }
    
    loadDatasets();
    
    // 检查元素是否存在
    if (datasetSelect && userSelect) {
        datasetSelect.addEventListener('change', function() {
            if (this.value) {
                loadUsers(this.value);
            } else {
                userSelect.innerHTML = '<option value="">请先选择数据集</option>';
                userSelect.disabled = true;
            }
        });
    }
    
    // 检查元素是否存在
    if (generatePathBtn) {
        generatePathBtn.addEventListener('click', generatePath);
    }
    
    async function loadDatasets() {
        // 检查元素是否存在
        if (!datasetSelect) return;
        
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
        // 检查元素是否存在
        if (!userSelect) return;
        
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
    
    async function generatePath() {
        // 检查元素是否存在
        if (!datasetSelect || !userSelect || !goalSelect || !pathLength || !generatePathBtn || !pathContainer || !pathCard) return;
        
        const dataset = datasetSelect.value;
        const userId = userSelect.value;
        const goal = goalSelect.value;
        const max_length = pathLength.value;
        
        if (!dataset || !userId) {
            alert('请选择数据集和学生');
            return;
        }
        
        generatePathBtn.disabled = true;
        generatePathBtn.textContent = '生成中...';
        
        try {
            const response = await fetch(`/api/path/${dataset}/${userId}/?goal=${goal}&max_length=${max_length}`);
            const data = await response.json();
            
            pathContainer.innerHTML = '';
            
            if (data.path && data.path.length > 0) {
                data.path.forEach((item, index) => {
                    const div = document.createElement('div');
                    div.className = 'path-step';
                    div.innerHTML = `
                        <div class="step-number">${item.step}</div>
                        <div class="step-content">
                            <div class="step-title">${item.skill_name}</div>
                            <div class="step-description">知识点 ID: ${item.skill_id}</div>
                        </div>
                        ${index < data.path.length - 1 ? '<div class="step-arrow">↓</div>' : ''}
                    `;
                    pathContainer.appendChild(div);
                });
            } else {
                pathContainer.innerHTML = '<p class="no-data">暂无学习路径建议</p>';
            }
            
            pathCard.style.display = 'block';
        } catch (error) {
            console.error('生成学习路径失败:', error);
            pathContainer.innerHTML = '<p class="error-message">生成学习路径失败</p>';
            pathCard.style.display = 'block';
        } finally {
            generatePathBtn.disabled = false;
            generatePathBtn.textContent = '生成学习路径';
        }
    }
});