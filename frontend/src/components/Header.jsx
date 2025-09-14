import React from 'react';
import { Activity, Target, Calendar, Trophy } from 'lucide-react';

const Header = ({ userName, weeklyProgress }) => {
  return (
    <header className="dashboard-header">
      <div className="dashboard-card">
        <div className="card-header">
          <h1 className="card-title">
            <Activity size={28} />
            Fitness Dashboard
          </h1>
          <div className="user-info">
            <span>Welcome, {userName}!</span>
          </div>
        </div>
        
        <div className="stats-grid">
          <div className="stat-card" style={{background: "linear-gradient(135deg, #ff00d4ff 0%, #f871e6ff 100%)"}}>
            <Target size={24} />
            <div className="stat-value">{weeklyProgress.currentCalories}</div>
            <div className="stat-label">Calories Burned</div>
          </div>
          
          <div className="stat-card" style={{background: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"}}>
            <Calendar size={24} />
            <div className="stat-value">{weeklyProgress.workoutsCompleted}/5</div>
            <div className="stat-label">Workouts Completed</div>
          </div>
          
          <div className="stat-card" style={{background: "linear-gradient(135deg, #43e97b 0%, #aff938ff 100%)"}}>
            <Trophy size={24} />
            <div className="stat-value">{weeklyProgress.achievementRate}%</div>
            <div className="stat-label">Weekly Goal</div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;