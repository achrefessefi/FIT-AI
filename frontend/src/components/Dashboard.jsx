import React from 'react';
import Header from './Header';
import WeeklyProgress from './WeeklyProgress';
import CalorieComparison from './CalorieComparison';
import PerformanceChart from './PerformanceChart';
import WorkoutSchedule from './WorkoutSchedule';
import CoachFeedback from './CoachFeedback';

const Dashboard = () => {
  // Sample data - replace with real data from your backend
  const userData = {
    userName: "Alex Johnson",
    weeklyProgress: {
      currentCalories: 2450,
      workoutsCompleted: 3,
      achievementRate: 75
    }
  };

  const weeklyData = {
    caloriesData: [320, 450, 510, 480, 690, 0, 0],
    targetData: [400, 400, 400, 400, 400, 400, 400]
  };

  const calorieData = {
    consumed: 2450,
    total: 3200
  };

  const performanceData = {
    scores: [85, 78, 92, 75, 82]
  };

  const scheduleData = [
    { time: "Today, 6:00 PM", name: "Upper Body Strength", type: "Strength" },
    { time: "Wed, 7:00 PM", name: "Interval Running", type: "Cardio" },
    { time: "Fri, 6:30 PM", name: "Full Body HIIT", type: "HIIT" },
    { time: "Sun, 9:00 AM", name: "Yoga & Stretching", type: "Yoga" }
  ];

  const coachFeedback = {
    message: "Excellent depth on your squats today! Your form has improved significantly. Keep focusing on maintaining a tight core during push-ups to maximize effectiveness.",
    strength: "Squat Form: 92%",
    improvement: "Push-up Stability: 78%"
  };

  return (
    <div className="dashboard-container">
      <Header 
        userName={userData.userName} 
        weeklyProgress={userData.weeklyProgress} 
      />
      
      <div className="dashboard-grid">
        <div className="grid-col-8">
          <WeeklyProgress data={weeklyData} />
        </div>
        
        <div className="grid-col-4">
          <CalorieComparison data={calorieData} />
        </div>
        
        <div className="grid-col-6">
          <PerformanceChart data={performanceData} />
        </div>
        
        <div className="grid-col-3">
          <WorkoutSchedule schedule={scheduleData} />
        </div>
        
        <div className="grid-col-3">
          <CoachFeedback feedback={coachFeedback} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;