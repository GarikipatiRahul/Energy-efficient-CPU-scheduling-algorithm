// Initialize charts
let cpuChart;
let memoryGauge;

// CPU Chart initialization
const initCPUChart = () => {
    const ctx = document.getElementById('cpuChart').getContext('2d');
    cpuChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(10).fill(''),
            datasets: [{
                label: 'CPU Usage %',
                data: Array(10).fill(0),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
};

// Memory gauge initialization
const initMemoryGauge = () => {
    const ctx = document.getElementById('memoryGauge').getContext('2d');
    memoryGauge = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Used', 'Free'],
            datasets: [{
                data: [0, 100],
                backgroundColor: ['#4f46e5', '#e5e7eb']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%'
        }
    });
};

// Update active tasks list
const updateActiveTasks = (tasks) => {
    const container = document.getElementById('activeTasks');
    container.innerHTML = '';
    
    tasks.forEach(task => {
        const taskElement = document.createElement('div');
        taskElement.className = 'task-item';
        taskElement.innerHTML = `
            <div class="font-medium">${task.name || 'Unknown Task'}</div>
            <div class="text-sm text-gray-600">
                Priority: ${task.priority || 'N/A'} | 
                CPU: ${task.cpu_usage || '0'}%
            </div>
        `;
        container.appendChild(taskElement);
    });
};

// Update scheduler status
const updateSchedulerStatus = (status) => {
    const container = document.getElementById('schedulerStatus');
    container.innerHTML = `
        <div class="flex items-center">
            <div class="w-3 h-3 rounded-full ${status === 'Running' ? 'bg-green-500' : 'bg-red-500'} mr-2"></div>
            <span>${status}</span>
        </div>
    `;
};

// Fetch and update dashboard data
const updateDashboard = async () => {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        // Update CPU chart
        cpuChart.data.datasets[0].data.shift();
        cpuChart.data.datasets[0].data.push(data.cpu_usage[0]);
        cpuChart.update();

        // Update memory gauge
        memoryGauge.data.datasets[0].data = [data.memory_usage, 100 - data.memory_usage];
        memoryGauge.update();

        // Update tasks and status
        updateActiveTasks(data.active_tasks);
        updateSchedulerStatus(data.scheduler_status);
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initCPUChart();
    initMemoryGauge();
    
    // Update every 2 seconds
    setInterval(updateDashboard, 2000);
});
