import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { BarChart3 } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const PerformanceChart = ({ data }) => {
  const chartData = {
    labels: ['Squats', 'Push-ups', 'Plank', 'Lunges', 'Burpees'],
    datasets: [
      {
        label: 'Performance Score',
        data: data.scores,
        backgroundColor: 'rgba(225, 252, 127, 0.8)',
        borderColor: 'rgba(216, 248, 175, 0.8)',
        borderWidth: 1,
        borderRadius: 5,
      },
    ],
  };

  const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        color: '#ffffff', // ✅ legend text color
      },
    },
    title: {
      display: true,
      text: 'Exercise Performance Scores',
      color: '#ffffff', // ✅ title color
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      max: 100,
      ticks: {
        color: '#ffffff', // ✅ Y-axis numbers color
      },
      grid: {
        color: 'rgba(255, 251, 251, 0.1)',
      },
    },
    x: {
      ticks: {
        color: '#ffffff', // ✅ X-axis labels color
      },
      grid: {
        display: false,
      },
    },
  },
};


  return (
    <div className="dashboard-card">
      <div className="card-header">
        <h2 className="card-title">
          <BarChart3 size={24} />
          Performance Metrics
        </h2>
      </div>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default PerformanceChart;