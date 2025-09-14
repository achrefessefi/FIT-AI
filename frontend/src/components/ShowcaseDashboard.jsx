import React, { useEffect, useState } from "react";
import Header from "./Header";
import WeeklyProgress from "./WeeklyProgress";
import CalorieComparison from "./CalorieComparison";
import PerformanceChart from "./PerformanceChart";
import CoachFeedback from "./CoachFeedback";
import { fetchDashboard, fetchPlan } from "../api";

export default function ShowcaseDashboard() {
  // ----- Static / mock for charts (kept for pitch)
  const userData = {
    userName: "Alex Johnson",
    weeklyProgress: { currentCalories: 2450, workoutsCompleted: 3, achievementRate: 75 },
  };
  const weeklyData = { caloriesData: [320, 450, 510, 480, 690, 0, 0], targetData: [400, 400, 400, 400, 400, 400, 400] };
  const calorieData = { consumed: 2450, total: 3200 };

  // Live overrides
  const [performanceData, setPerformanceData] = useState({ scores: [85, 78, 92, 75, 82] });
  const [coachFeedback, setCoachFeedback] = useState({
    message: "Excellent depth on your squats today! Your form has improved.",
    strength: "Squat Form: 92%",
    improvement: "Push-up Stability: 78%",
  });
  
  const [liveCoachNote, setLiveCoachNote] = useState(false);

  // NEW: Today cards state (from dashboard.json)
  const [todaySquat, setTodaySquat] = useState(null);
  const [todayPush, setTodayPush] = useState(null);

  // NEW: Upcoming pulls from persisted /api/plan
  const [upcoming, setUpcoming] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        // 1) dashboard.json (today)
        const d = await fetchDashboard();
        const squat = d?.summaries?.squat || {};
        const push = d?.summaries?.push || {};

        // Performance first two bars
        const squatScore =
          typeof squat?.metrics?.good_rep_rate_pct === "number"
            ? Math.max(0, Math.min(100, squat.metrics.good_rep_rate_pct))
            : 85;
        const pushScore =
          typeof push?.quality?.final_score === "number"
            ? Math.max(0, Math.min(100, push.quality.final_score))
            : 78;

        setPerformanceData((old) => ({ scores: [squatScore, pushScore, old.scores[2], old.scores[3], old.scores[4]] }));

        // Coach feedback (prefer push LLM text, else squat cue/explanation)
        const liveMsg =
          push?.last_set_summary_text ||
          (Array.isArray(squat?.coaching_cues) && squat.coaching_cues[0]) ||
          (Array.isArray(squat?.explanations) && squat.explanations[0]) ||
          "Great work. Keep a steady cadence and full range.";

        const liveStrength = `Squat good reps: ${squat?.totals?.good_reps_total ?? "—"}/${squat?.totals?.reps_completed ?? "—"}`;
        const liveImprove = `Push label: ${push?.quality?.final_label || "—"}`;

        setCoachFeedback({ message: liveMsg, strength: liveStrength, improvement: liveImprove });
        setTodaySquat(squat);
        setTodayPush(push);

        
        setLiveCoachNote(true);
      } catch { /* keep mock visuals */ }

      try {
        // 2) plan.json (upcoming)
        const plan = await fetchPlan();
        // order the saved plan by week-day order if needed
        const order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
        const s = Array.isArray(plan?.sessions) ? [...plan.sessions] : [];
        s.sort((a, b) => order.indexOf((a?.day || "").toLowerCase()) - order.indexOf((b?.day || "").toLowerCase()));
        setUpcoming(s);
      } catch { /* if no plan yet, keep empty; card will show hint */ }
    })();
  }, []);

  // helper to pretty a session row
  const prettySession = (s) =>
    `${(s?.day || "").toUpperCase()}, ${s?.type || "—"} • ${s?.duration ?? "—"} min • ` +
    (Array.isArray(s?.exercises) && s.exercises.length
      ? s.exercises.map((e) => `${e.name} ${e.sets}×${e.reps}`).join(" · ")
      : "—");

  // pull a few “today” numbers safely
  const squatKPI = todaySquat?.metrics || {};
  const squatTotals = todaySquat?.totals || {};
  const pushQuality = todayPush?.quality || {};

  return (
    <div className="dashboard-container">
      <Header userName={userData.userName} weeklyProgress={userData.weeklyProgress} />

      <div className="dashboard-grid">
        <div className="grid-col-8">
          <WeeklyProgress data={weeklyData} />
        </div>

        <div className="grid-col-4">
          <div className="dashboard-card" style={{ position: "relative" }}>
            <CalorieComparison data={calorieData} />
          </div>
        </div>

        <div className="grid-col-6">
          
          <PerformanceChart data={performanceData} />
        </div>

        {/* === NEW: Today cards (from dashboard.json) ===================== */}
        <div className="grid-col-3">
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-title">Today · Squat KPIs</div>
            </div>
            {!todaySquat ? (
              <div className="hint">No session yet.</div>
            ) : (
              <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.6 }}>
                <li>Good reps: <b>{squatTotals.good_reps_total ?? "—"}</b> / {squatTotals.reps_completed ?? "—"}</li>
                <li>Good-rep rate: <b>{typeof squatKPI.good_rep_rate_pct === "number" ? `${squatKPI.good_rep_rate_pct.toFixed(1)}%` : "—"}</b></li>
                <li>Avg cadence: {squatKPI.avg_cadence_rpm ?? "—"} rpm</li>
                <li>Up speed (norm): {squatKPI.avg_up_speed_norm ?? "—"}</li>
                <li>Down speed: {squatKPI.avg_down_speed_deg_s ?? "—"}°/s</li>
                <li>Motivation: {squatKPI.avg_motivation ?? "—"}</li>
              </ul>
            )}
          </div>
        </div>

        <div className="grid-col-3">
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-title">Today · Push Summary</div>
            </div>
            {!todayPush ? (
              <div className="hint">No session yet.</div>
            ) : (
              <div style={{ display: "grid", gap: 8 }}>
                <div>Score: <b>{pushQuality.final_score ?? "—"}</b> ({pushQuality.final_label || "—"})</div>
                {Array.isArray(pushQuality.set_scores) && (
                  <div className="hint">Sets: {pushQuality.set_scores.join(" • ")}</div>
                )}
                <div className="hr" />
                <div className="hint" style={{ whiteSpace: "pre-wrap" }}>
                  {(todayPush.last_set_summary_text || "").trim()}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* === Upcoming Workouts from persisted plan ====================== */}
        <div className="grid-col-12">
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-title">Upcoming Workouts (from generated plan)</div>
            </div>
            {!upcoming.length ? (
              <div className="hint">No plan found. Go to the main Dashboard and click “Generate plan for me”.</div>
            ) : (
              <div className="workout-schedule">
                {upcoming.map((s, i) => (
                  <div key={i} className="schedule-item">
                    <div>
                      <div className="schedule-time">{(s.day || "").toUpperCase()}</div>
                      <div>{prettySession(s)}</div>
                    </div>
                    <div className={`schedule-type type-${(s.type || "strength").toLowerCase()}`}>
                      {s.type || "Strength"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Coach feedback (uses live text already) */}
        <div className="grid-col-12">
          <div className="dashboard-card">
            <CoachFeedback feedback={coachFeedback} />
            {liveCoachNote && <div className="hint" style={{ marginTop: 8 }}>* Using latest first-session summary.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
