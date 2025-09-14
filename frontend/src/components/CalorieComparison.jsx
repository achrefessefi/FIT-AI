import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { PieChart } from 'lucide-react';

ChartJS.register(ArcElement, Tooltip, Legend);

const CalorieComparison = ({ data }) => {
  const chartData = {
    labels: ['Completed', 'Remaining'],
    datasets: [
      {
        data: [data.consumed, data.total - data.consumed],
        backgroundColor: [
          'rgba(248, 247, 250, 0.8)',
          'rgba(252, 252, 252, 0.2)',
        ],
        borderColor: [
          'rgba(241, 253, 253, 1)',
          'rgba(255, 252, 99, 0.2)',
        ],
        borderWidth: 1,
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
      text: 'Weekly Calorie Goal Progress',
      color: '#ffffff', // ✅ title color
    },
  },
  cutout: '70%',
};


  return (
    <div className="dashboard-card">
      <div className="card-header">
        <h2 className="card-title">
          <PieChart size={24} />
          Calorie Progress
        </h2>
      </div>
      <div style={{ position: 'relative', height: '300px' }}>
        <Doughnut data={chartData} options={options} />
        <div style={{
          position: 'absolute',
          top: '60%',
          left: '40%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          fontSize: '1.5rem',
          fontWeight: 'bold'
        }}>
          {Math.round((data.consumed / data.total) * 100)}%
        </div>
      </div>
    </div>
  );
};

export default CalorieComparison;