import React from 'react';
import { Calendar, Clock, Activity } from 'lucide-react';

const WorkoutSchedule = ({ schedule }) => {
  const getTypeClass = (type) => {
    switch (type.toLowerCase()) {
      case 'strength': return 'type-strength';
      case 'cardio': return 'type-cardio';
      case 'yoga': return 'type-yoga';
      case 'hiit': return 'type-hiit';
      default: return 'type-strength';
    }
  };

  return (
    <div className="dashboard-card">
      <div className="card-header">
        <h2 className="card-title">
          <Calendar size={24} />
          Upcoming Workouts
        </h2>
      </div>
      
      <div className="workout-schedule">
        {schedule.map((item, index) => (
          <div key={index} className="schedule-item">
            <div>
              <div className="schedule-time">
                <Clock size={16} style={{ display: 'inline', marginRight: '8px' }} />
                {item.time}
              </div>
              <div>{item.name}</div>
            </div>
            <div className={`schedule-type ${getTypeClass(item.type)}`}>
              <Activity size={14} style={{ display: 'inline', marginRight: '4px' }} />
              {item.type}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default WorkoutSchedule;