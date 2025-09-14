import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const WeeklyProgress = ({ data }) => {
  const chartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Calories Burned',
        data: data.caloriesData,
        borderColor: 'rgba(242, 254, 79, 1)',
        backgroundColor: 'rgba(79, 172, 254, 0.1)',
        fill: true,
        tension: 0.4,
        pointBackgroundColor: 'rgba(255, 2, 200, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgb(79, 172, 254)',
      },
      {
        label: 'Calories Target',
        data: data.targetData,
        borderColor: 'rgba(252, 252, 252, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: true,
        tension: 0.4,
        borderDash: [5, 5],
        pointStyle: false,
      }
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
      text: 'Weekly Calorie Progress',
      color: '#ffffff', // ✅ title color
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      ticks: {
        color: '#ffffff', // ✅ Y-axis numbers color
      },
      grid: {
        color: 'rgba(0, 0, 0, 0.1)',
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
          <TrendingUp size={24} />
          Weekly Progress
        </h2>
      </div>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default WeeklyProgress;