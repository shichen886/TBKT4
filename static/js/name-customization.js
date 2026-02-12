document.addEventListener('DOMContentLoaded', function() {
    const nameType = document.getElementById('nameType');
    const datasetSelect = document.getElementById('datasetSelect');
    const idSelect = document.getElementById('idSelect');
    const currentName = document.getElementById('currentName');
    const newName = document.getElementById('newName');
    const saveNameBtn = document.getElementById('saveNameBtn');
    const autoGenerateBtn = document.getElementById('autoGenerateBtn');
    const batchGenerateBtn = document.getElementById('batchGenerateBtn');
    const nameForm = document.getElementById('nameForm');
    const statsSection = document.getElementById('statsSection');
    const studentCount = document.getElementById('studentCount');
    const skillCount = document.getElementById('skillCount');
    const itemCount = document.getElementById('itemCount');
    
    let currentDataset = '';
    let currentMappings = {};
    
    // 加载数据集列表
    loadDatasets();
    
    // 监听数据集选择变化
    datasetSelect.addEventListener('change', function() {
        currentDataset = this.value;
        if (currentDataset) {
            loadMappings();
            loadIds();
            nameForm.style.display = 'block';
            statsSection.style.display = 'block';
        } else {
            nameForm.style.display = 'none';
            statsSection.style.display = 'none';
        }
    });
    
    // 监听名称类型变化
    nameType.addEventListener('change', function() {
        if (currentDataset) {
            loadIds();
        }
        
        // 显示或隐藏批量生成按钮
        if (this.value === 'skill') {
            batchGenerateBtn.style.display = 'inline-block';
        } else {
            batchGenerateBtn.style.display = 'none';
        }
    });
    
    // 监听ID选择变化
    idSelect.addEventListener('change', function() {
        if (this.value) {
            loadCurrentName(this.value);
        }
    });
    
    // 保存名称
    saveNameBtn.addEventListener('click', async function() {
        const id = idSelect.value;
        const name = newName.value.trim();
        
        if (!id) {
            alert('请选择ID');
            return;
        }
        
        if (!name) {
            alert('请输入新名称');
            return;
        }
        
        try {
            const response = await fetch('/api/save-name/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset: currentDataset,
                    type: nameType.value,
                    id: parseInt(id),
                    name: name
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('✅ 名称已保存！');
                loadMappings();
                loadCurrentName(id);
            } else {
                alert('❌ 保存失败：' + (data.error || '未知错误'));
            }
        } catch (error) {
            console.error('保存名称失败:', error);
            alert('❌ 保存失败，请检查网络连接');
        }
    });
    
    // 自动生成名称
    autoGenerateBtn.addEventListener('click', async function() {
        const id = idSelect.value;
        
        if (!id) {
            alert('请选择ID');
            return;
        }
        
        try {
            const response = await fetch('/api/auto-generate-name/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset: currentDataset,
                    type: nameType.value,
                    id: parseInt(id)
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('✅ 已生成默认名称！');
                loadMappings();
                loadCurrentName(id);
            } else {
                alert('❌ 生成失败：' + (data.error || '未知错误'));
            }
        } catch (error) {
            console.error('生成名称失败:', error);
            alert('❌ 生成失败，请检查网络连接');
        }
    });
    
    // 批量生成名称
    batchGenerateBtn.addEventListener('click', async function() {
        if (!confirm('确定要为所有知识点生成默认名称吗？')) {
            return;
        }
        
        try {
            const response = await fetch('/api/batch-generate-names/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset: currentDataset,
                    type: nameType.value
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('✅ 已为所有知识点生成名称！');
                loadMappings();
                loadIds();
            } else {
                alert('❌ 生成失败：' + (data.error || '未知错误'));
            }
        } catch (error) {
            console.error('批量生成名称失败:', error);
            alert('❌ 生成失败，请检查网络连接');
        }
    });
    
    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets/');
            const data = await response.json();
            
            if (data.success) {
                datasetSelect.innerHTML = '<option value="">请选择数据集...</option>';
                data.datasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset;
                    option.textContent = dataset;
                    datasetSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('加载数据集失败:', error);
        }
    }
    
    async function loadMappings() {
        try {
            const response = await fetch(`/api/mappings/${currentDataset}/`);
            const data = await response.json();
            
            if (data.success) {
                currentMappings = data.mappings;
                updateStats();
            }
        } catch (error) {
            console.error('加载映射失败:', error);
        }
    }
    
    async function loadIds() {
        try {
            const response = await fetch(`/api/ids/${currentDataset}/${nameType.value}/`);
            const data = await response.json();
            
            if (data.success) {
                idSelect.innerHTML = '<option value="">请选择...</option>';
                data.ids.forEach(id => {
                    const option = document.createElement('option');
                    option.value = id;
                    option.textContent = `${getDisplayName(id)} (ID: ${id})`;
                    idSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('加载ID列表失败:', error);
        }
    }
    
    function loadCurrentName(id) {
        const name = getDisplayName(id);
        currentName.value = name;
        newName.value = name;
    }
    
    function getDisplayName(id) {
        const type = nameType.value;
        if (type === 'student') {
            return currentMappings.user_names && currentMappings.user_names[id] || `学生${id}`;
        } else if (type === 'skill') {
            return currentMappings.skill_names && currentMappings.skill_names[id] || `知识点${id}`;
        } else if (type === 'item') {
            return currentMappings.item_names && currentMappings.item_names[id] || `题目${id}`;
        }
        return `ID: ${id}`;
    }
    
    function updateStats() {
        studentCount.textContent = currentMappings.user_names ? Object.keys(currentMappings.user_names).length : 0;
        skillCount.textContent = currentMappings.skill_names ? Object.keys(currentMappings.skill_names).length : 0;
        itemCount.textContent = currentMappings.item_names ? Object.keys(currentMappings.item_names).length : 0;
    }
});