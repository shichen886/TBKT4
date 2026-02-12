document.addEventListener('DOMContentLoaded', function() {
    const datasetSelect = document.getElementById('datasetSelect');
    const userSelect = document.getElementById('userSelect');
    const analysisType = document.getElementById('analysisType');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const chartCard = document.getElementById('chartCard');
    const chartTitle = document.getElementById('chartTitle');
    const chartContainer = document.getElementById('chartContainer');
    const weakSkillsCard = document.getElementById('weakSkillsCard');
    const weakSkillsList = document.getElementById('weakSkillsList');
    
    loadDatasets();
    
    datasetSelect.addEventListener('change', function() {
        if (this.value) {
            loadUsers(this.value);
        } else {
            userSelect.innerHTML = '<option value="">请先选择数据集</option>';
            userSelect.disabled = true;
        }
    });
    
    analyzeBtn.addEventListener('click', performAnalysis);
    
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
    
    async function performAnalysis() {
        const dataset = datasetSelect.value;
        const userId = userSelect.value;
        const type = analysisType.value;
        
        if (!dataset || !userId) {
            alert('请选择数据集和学生');
            return;
        }
        
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '分析中...';
        
        try {
            if (type === 'skills') {
                await analyzeSkills(dataset, userId);
            } else if (type === 'trend') {
                await analyzeTrend(dataset, userId);
            } else if (type === 'errors') {
                await analyzeErrors(dataset, userId);
            }
        } catch (error) {
            console.error('分析失败:', error);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '开始分析';
        }
    }
    
    async function analyzeSkills(dataset, userId) {
        try {
            const response = await fetch(`/api/skills/${dataset}/${userId}/`);
            const data = await response.json();
            
            chartTitle.textContent = '各知识点正确率';
            
            const skills = data.skills.map(s => s.skill_name);
            const accuracies = data.skills.map(s => s.accuracy * 100);
            
            const trace = {
                x: skills,
                y: accuracies,
                type: 'bar',
                marker: {
                    color: accuracies.map(a => a >= 70 ? '#10B981' : a >= 50 ? '#F59E0B' : '#EF4444'),
                    line: {
                        color: '#ffffff',
                        width: 1
                    }
                }
            };
            
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.1)',
                font: {
                    color: '#ffffff'
                },
                xaxis: {
                    title: '知识点',
                    gridcolor: 'rgba(255,255,255,0.2)'
                },
                yaxis: {
                    title: '正确率 (%)',
                    gridcolor: 'rgba(255,255,255,0.2)',
                    range: [0, 100]
                },
                margin: {
                    l: 60,
                    r: 20,
                    t: 40,
                    b: 120
                }
            };
            
            Plotly.newPlot(chartContainer, [trace], layout, {responsive: true});
            chartCard.style.display = 'block';
            
            weakSkillsList.innerHTML = '';
            const weakSkills = data.skills.filter(s => s.accuracy < 0.7).slice(0, 5);
            
            if (weakSkills.length > 0) {
                weakSkills.forEach(skill => {
                    const div = document.createElement('div');
                    div.className = 'weak-skill-item';
                    div.innerHTML = `
                        <span class="skill-name">${skill.skill_name}</span>
                        <div class="skill-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${skill.accuracy * 100}%"></div>
                            </div>
                            <span class="progress-text">${(skill.accuracy * 100).toFixed(2)}%</span>
                        </div>
                    `;
                    weakSkillsList.appendChild(div);
                });
                weakSkillsCard.style.display = 'block';
            } else {
                weakSkillsCard.style.display = 'none';
            }
        } catch (error) {
            console.error('知识点分析失败:', error);
        }
    }
    
    async function analyzeTrend(dataset, userId) {
        try {
            const response = await fetch(`/api/trend/${dataset}/${userId}/`);
            const data = await response.json();
            
            chartTitle.textContent = '累计正确率趋势';
            weakSkillsCard.style.display = 'none';
            
            const x = data.trend.map(t => t.index + 1);
            const y = data.trend.map(t => t.cumulative_accuracy * 100);
            
            const trace = {
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines+markers',
                line: {
                    color: '#4F46E5',
                    width: 3
                },
                marker: {
                    color: '#4F46E5',
                    size: 6
                }
            };
            
            const shapes = [{
                type: 'line',
                x0: 0,
                x1: Math.max(...x),
                y0: 70,
                y1: 70,
                line: {
                    color: '#EF4444',
                    width: 2,
                    dash: 'dash'
                }
            }];
            
            const annotations = [{
                x: Math.max(...x),
                y: 70,
                text: '目标线 70%',
                showarrow: false,
                xanchor: 'right',
                yanchor: 'bottom',
                font: {
                    color: '#EF4444'
                }
            }];
            
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.1)',
                font: {
                    color: '#ffffff'
                },
                xaxis: {
                    title: '答题序号',
                    gridcolor: 'rgba(255,255,255,0.2)'
                },
                yaxis: {
                    title: '累计正确率 (%)',
                    gridcolor: 'rgba(255,255,255,0.2)',
                    range: [0, 100]
                },
                margin: {
                    l: 60,
                    r: 20,
                    t: 40,
                    b: 60
                },
                shapes: shapes,
                annotations: annotations
            };
            
            Plotly.newPlot(chartContainer, [trace], layout, {responsive: true});
            chartCard.style.display = 'block';
        } catch (error) {
            console.error('趋势分析失败:', error);
        }
    }
    
    async function analyzeErrors(dataset, userId) {
        try {
            const response = await fetch(`/api/errors/${dataset}/${userId}/`);
            const data = await response.json();
            
            chartTitle.textContent = '错误分布';
            weakSkillsCard.style.display = 'none';
            
            if (data.errors.length > 0) {
                const labels = data.errors.map(e => e.skill_name);
                const values = data.errors.map(e => e.count);
                
                const trace = {
                    labels: labels,
                    values: values,
                    type: 'pie',
                    marker: {
                        colors: [
                            '#4F46E5', '#818CF8', '#A78BFA', '#C4B5FD',
                            '#F97316', '#FB923C', '#FDBA74', '#FED7AA',
                            '#10B981', '#34D399'
                        ]
                    },
                    textinfo: 'label+percent',
                    textposition: 'inside'
                };
                
                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        color: '#ffffff'
                    },
                    margin: {
                        l: 20,
                        r: 20,
                        t: 40,
                        b: 20
                    }
                };
                
                Plotly.newPlot(chartContainer, [trace], layout, {responsive: true});
                chartCard.style.display = 'block';
                
                weakSkillsCard.style.display = 'block';
                weakSkillsList.innerHTML = '<h4>需要重点复习的知识点</h4>';
                data.errors.forEach(error => {
                    const div = document.createElement('div');
                    div.className = 'error-item';
                    div.innerHTML = `
                        <span class="error-skill">${error.skill_name}</span>
                        <span class="error-count">${error.count} 次错误</span>
                    `;
                    weakSkillsList.appendChild(div);
                });
            } else {
                chartContainer.innerHTML = '<p class="success-message">🎉 恭喜！该学生没有错误记录</p>';
                chartCard.style.display = 'block';
                weakSkillsCard.style.display = 'none';
            }
        } catch (error) {
            console.error('错误分析失败:', error);
        }
    }
});