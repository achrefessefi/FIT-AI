import React from 'react';
import { MessageCircle, Award, Target } from 'lucide-react';

const CoachFeedback = ({ feedback }) => {
  return (
    <div className="dashboard-card">
      <div className="card-header">
        <h2 className="card-title">
          <MessageCircle size={24} />
          Coach's Feedback
        </h2>
      </div>
      
      <div className="coach-feedback">
        <div className="feedback-text">
          "{feedback.message}"
        </div>
        
        <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Award size={16} />
            <span className="feedback-highlight">{feedback.strength}</span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Target size={16} />
            <span className="feedback-highlight">{feedback.improvement}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CoachFeedback;