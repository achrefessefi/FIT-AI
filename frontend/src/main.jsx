// frontend/src/main.jsx
import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Link, useNavigate } from "react-router-dom";
import { FaDumbbell, FaRunning } from "react-icons/fa";
import { GiLotus, GiMuscleUp } from "react-icons/gi";
import ShowcaseDashboard from "./components/ShowcaseDashboard";

// --- visual padding/typography bump ---
function injectCSS(css) {
  const style = document.createElement("style");
  style.setAttribute("data-ui-bump", "true");
  style.textContent = css;
  document.head.appendChild(style);
}

injectCSS(`
  /* Give 3-up grids more breathing room + wider cards */
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 24px;
  }
  @media (min-width: 1280px) {
    .grid-3 {
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 28px;
    }
  }

  /* Card sizing */
  .plan-tile {
    padding: 22px !important;
    border-radius: 16px;
  }

  /* Titles bigger & clearer */
  .plan-tile .title {
    font-size: 1.18rem;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: 0.2px;
  }

  /* Body text spacing */
  .plan-tile .meta {
    font-size: 0.98rem;
    line-height: 1.55;
  }

  /* Chips/badges slightly larger */
  .chip, .badge {
    font-size: 0.95rem;
    padding: 6px 10px;
    border-radius: 999px;
  }

  /* Divider more subtle, with spacing */
  .plan-tile .hr {
    height: 1px;
    background: rgba(0,0,0,0.08);
    margin: 12px 0;
  }

  /* Meal calendar tiles a bit taller so lines don't feel cramped */
  .meal-calendar .plan-tile {
    min-height: 190px;
  }
  .meal-calendar .plan-tile .title {
    font-size: 1.05rem;
  }

  /* Card titles slightly bigger app-wide */
  .card .card-title {
    font-size: 1.15rem;
    line-height: 1.3;
  }

  /* Buttons on tiles: give fingers room */
  .btn.small {
    padding: 8px 12px;
    font-size: 0.95rem;
  }

  /* ---- NEW: exercise pills wrapping cleanly ---- */
  .exercise-badges {
    display: flex;
    flex-wrap: wrap;       /* wrap only between badges */
    gap: 10px;             /* space between pills */
    align-items: center;
    margin-top: 6px;
  }
  .badge--exercise {
    display: inline-flex;  /* keeps icon + text together */
    align-items: center;
    white-space: nowrap;   /* never break inside the pill */
    word-break: keep-all;  /* extra safety */
    line-height: 1;        /* compact vertical rhythm */
    padding: 8px 12px;     /* comfy */
    border-radius: 999px;
  }
  .badge--exercise svg {
    margin-right: 8px;
    flex: 0 0 auto;
  }
`);

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const cls = (...a) => a.filter(Boolean).join(" ");
const orderDays = ["mon","tue","wed","thu","fri","sat","sun"];

/* ------------------------------------------------
   Utilities
-------------------------------------------------*/
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const capitalize = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

const three = (s) => s.trim().toLowerCase().slice(0, 3);
const day3ToNice = { mon: "Mon", tue: "Tue", wed: "Wed", thu: "Thu", fri: "Fri", sat: "Sat", sun: "Sun" };

const toNiceDaysString = (arr) =>
  Array.isArray(arr) && arr.length
    ? arr.map((d) => day3ToNice[d] || day3ToNice[three(d)]).filter(Boolean).join(", ")
    : "Mon, Wed, Fri";

const toDay3Array = (str) => {
  if (!str) return ["mon", "wed", "fri"];
  const tokens = str.split(/[,\s]+/).filter(Boolean);
  const out = [];
  for (const t of tokens) {
    const key = three(t);
    if (day3ToNice[key]) {
      if (!out.includes(key)) out.push(key);
    }
  }
  return out.length ? out : ["mon", "wed", "fri"];
};

// --- exercise helpers for finding Squats/Push-ups on a card ---
const _slug = (s) => (s || "").toLowerCase().replace(/[^a-z]/g, "");
const findExercise = (card, needleSlug) =>
  (Array.isArray(card?.exercises) ? card.exercises : []).find(
    (e) => _slug(e?.name).includes(needleSlug)
  );


/* ------------------------------------------------
   Parsers for voice
-------------------------------------------------*/
const parseName = (t) => {
  const m = t.match(/(?:my name is|name is|it's|i am|i'm)\s+([a-zA-Z]+)/);
  return m ? capitalize(m[1]) : capitalize(t.trim().split(/\s+/)[0] || "");
};
const parseNumber = (t) => {
  const m = t.match(/(\d{1,3})/);
  return m ? +m[1] : null;
};
const parseStyle = (t) => {
  if (/\bhiit\b/.test(t)) return "hiit";
  if (/\bcardio\b/.test(t)) return "cardio";
  if (/\byoga\b/.test(t)) return "yoga";
  if (/\bstrength|weights?|resistance\b/.test(t)) return "strength";
  return null;
};
const parseGender = (t) => {
  const s = (t || "").toLowerCase();
  if (/\b(female|woman|girl)\b/.test(s)) return "female";
  if (/\b(male|man|boy)\b/.test(s)) return "male";
  if (/\b(non.?binary|nonbinary)\b/.test(s)) return "nonbinary";
  if (/\b(unspecified|skip|rather not say)\b/.test(s)) return "unspecified";
  return null;
};
const parseFitness = (t) => {
  const s = (t || "").toLowerCase();
  if (/\bbeginner|new\b/.test(s)) return "beginner";
  if (/\b(intermediate|average|okay)\b/.test(s)) return "intermediate";
  if (/\b(advanced|expert|pro)\b/.test(s)) return "expert";
  return null;
};
const parseGoal = (t) => {
  const s = (t || "").toLowerCase().trim();

  // lose weight / cutting
  if (/\b(?:lose|loose)\s*weight\b|\b(?:fat\s*loss|cut|cutting)\b/.test(s))
    return "lose_weight";

  // get fit / general fitness
  if (/\b(?:get\s*)?fit\b|\boverall\s*fitness\b|\bhealth\b|\btone\s*up\b/.test(s))
    return "get_fit";

  // gain muscle / get stronger / hypertrophy
  if (/\b(?:gain|build)(?:\s*lean)?\s*muscle\b|\bhypertrophy\b|\bbulk(?:ing)?\b|\b(?:get\s*)?strong(?:er)?\b|\bstrength\b/.test(s))
    return "gain_muscle";

  // flexibility / mobility
  if (/\bflex(?:ible|ibility)\b|\bmobility\b|\bmore\s*flexible\b/.test(s))
    return "more_flexible";

  return null;
};

const parseDays = (t) => {
  const text = (t || "").toLowerCase();

  if (/\b(keep(?:\s*it)?|same|no\s*change)\b/.test(text)) return "__KEEP__";

  const picked = [];
  const push = (lab) => !picked.includes(lab) && picked.push(lab);
  if (/\b(monday|mon)\b/.test(text)) push("Mon");
  if (/\b(tuesday|tue|tues)\b/.test(text)) push("Tue");
  if (/\b(wednesday|wed|weds)\b/.test(text)) push("Wed");
  if (/\b(thursday|thu|thur|thurs)\b/.test(text)) push("Thu");
  if (/\b(friday|fri)\b/.test(text)) push("Fri");
  if (/\b(saturday|sat)\b/.test(text)) push("Sat");
  if (/\b(sunday|sun)\b/.test(text)) push("Sun");
  if (picked.length) return picked.join(", ");

  const map = { mon: "Mon", tue: "Tue", wed: "Wed", thu: "Thu", fri: "Fri", sat: "Sat", sun: "Sun" };
  const tokens = text.split(/[,\s]+/).filter(Boolean).map((s) => s.slice(0, 3));
  const mapped = [];
  for (const tok of tokens) {
    const m = map[tok];
    if (m && !mapped.includes(m)) mapped.push(m);
  }
  return mapped.length ? mapped.join(", ") : null;
};
const parseTimeOfDay = (t) => {
  if (/\bmorning\b/.test(t)) return "morning";
  if (/\b(afternoon|midday)\b/.test(t)) return "afternoon";
  if (/\b(evening|night)\b/.test(t)) return "evening";
  return null;
};

const iconForStyle = (style) => {
  switch ((style || "").toLowerCase()) {
    case "cardio": return <FaRunning />;
    case "hiit": return <GiMuscleUp />;
    case "yoga": return <GiLotus />;
    case "strength": return <FaDumbbell />;
    default: return <FaDumbbell />;
  }
};

const ExerciseBadge = ({ name }) => (
  <span className="badge badge--exercise">
    <FaDumbbell style={{ verticalAlign: "-10%" }} /> {name}
  </span>
);

/* ------------------------------------------------
   Web Speech: TTS (SpeechSynthesis)
-------------------------------------------------*/
function useTTS() {
  const [supported, setSupported] = useState(false);
  useEffect(() => setSupported("speechSynthesis" in window), []);
  function speak(text, { onend } = {}) {
    if (!supported) return;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 1;
      u.pitch = 1;
      u.onend = () => onend && onend();
      window.speechSynthesis.speak(u);
    } catch {}
  }
  function cancel() {
    try { window.speechSynthesis.cancel(); } catch {}
  }
  return { supported, speak, cancel };
}

/* ------------------------------------------------
   Web Speech: STT (SpeechRecognition)
-------------------------------------------------*/
function useSTT(onFinal) {
  const recRef = useRef(null);
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = false;

    rec.onresult = (e) => {
      let finalText = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalText += t + " ";
        else setInterim(t);
      }
      if (finalText.trim()) {
        setInterim("");
        onFinal(finalText.toLowerCase());
      }
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);

    recRef.current = rec;
    setSupported(true);
  }, [onFinal]);

  const start = () => {
    if (!recRef.current) return;
    try { recRef.current.start(); setListening(true); } catch {}
  };
  const stop = () => {
    if (!recRef.current) return;
    try { recRef.current.stop(); } catch {}
    setListening(false);
  };

  return { supported, listening, interim, start, stop };
}

/* ------------------------------------------------
   Voice Wizard – no auto-save
-------------------------------------------------*/
function useVoiceWizard(initialForm, onApply) {
  const steps = useRef([
    { key: "name",           question: "What is your name?",                                                     parse: (t) => parseName(t),          confirm: (v) => `Got it. Hello ${v}.` },
    { key: "gender",         question: "What is your gender? female, male, or nonbinary?",                       parse: parseGender,                   confirm: (v) => `Gender set to ${v}.` },
    { key: "age",            question: "How old are you?",                                                       parse: (t) => clamp(parseNumber(t) ?? 0, 10, 90), confirm: (v) => `Age set to ${v} years.` },
    { key: "height",         question: "What is your height in centimeters?",                                    parse: (t) => clamp(parseNumber(t) ?? 0, 120, 230), confirm: (v) => `Height set to ${v} centimeters.` },
    { key: "weight",         question: "What is your weight in kilograms?",                                      parse: (t) => clamp(parseNumber(t) ?? 0, 30, 250),  confirm: (v) => `Weight set to ${v} kilograms.` },
    { key: "goal",           question: "What is your main goal: lose weight, get fit, get stronger, gain muscle or be more flexible?", parse: parseGoal, confirm: (v) => `Goal set to ${v.replace('_',' ')}.` },
    { key: "style",          question: "Which style do you prefer: cardio, H I I T, yoga, or strength?",         parse: parseStyle,                    confirm: (v) => `Preferred style set to ${v}.` },
    { key: "fitness_level",  question: "What is your fitness level: beginner, intermediate, or expert?",         parse: parseFitness,                  confirm: (v) => `Fitness level set to ${v}.` },
    { key: "days",           question: "Which days do you want to train? You can say keep it.",                  parse: parseDays,                     confirm: (v) => `Training days set to ${v}.` },
    { key: "tod",            question: "What time of day do you prefer: morning, afternoon, or evening?",        parse: parseTimeOfDay,                confirm: (v) => `Time of day set to ${v}.` },
    { key: "mins",           question: "How many minutes per session?",                                          parse: (t) => clamp(parseNumber(t) ?? 0, 10, 120), confirm: (v) => `Session length set to ${v} minutes.` },
  ]);

  const [form, setForm] = useState(initialForm);
  const [i, setI] = useState(0);
  const [active, setActive] = useState(false);
  const { supported: ttsOK, speak, cancel } = useTTS();
  const { supported: sttOK, listening, interim, start, stop } = useSTT(handleFinal);

  function handleFinal(text) {
    const step = steps.current[i];
    const parsed = step?.parse?.(text);

    if (parsed === null || parsed === undefined || parsed === "" || Number.isNaN(parsed)) {
      speak("Sorry, I didn't hear you clearly. Please say it again.", { onend: () => start() });
      return;
    }

    if (step.key === "days" && parsed === "__KEEP__") {
      speak(`Keeping your current training days: ${form.days}.`, { onend: () => nextStep() });
      return;
    }

    const newForm = { ...form, [step.key]: parsed };
    setForm(newForm);
    onApply(newForm);

    speak(step.confirm(parsed), { onend: () => nextStep() });
  }

  function nextStep() {
    const next = i + 1;
    if (next < steps.current.length) {
      setI(next);
      setTimeout(() => {
        const step = steps.current[next];
        speak(step.question, { onend: () => start() });
      }, 250);
    } else {
      setActive(false);
      stop();
      speak("All set. Review your details and press Save when you're ready.");
    }
  }

  function startWizard() {
    if (!ttsOK || !sttOK) return;
    setActive(true);
    setI(0);
    cancel();
    setTimeout(
      () =>
        speak("Okay, let's set up your profile. We will go one by one.", {
          onend: () => speak(steps.current[0].question, { onend: () => start() }),
        }),
      200
    );
  }

  function stopWizard() {
    setActive(false);
    stop();
    cancel();
  }

  useEffect(() => () => { stop(); cancel(); }, []);

  return { supported: ttsOK && sttOK, active, listening, interim, form, startWizard, stopWizard };
}

/* ------------------------------------------------
   Home (welcome + profile form + Voice Onboarding)
-------------------------------------------------*/
function Home() {
  const navigate = useNavigate();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(true);

  const [form, setForm] = useState({
    name: "Athlete",
    gender: "unspecified",
    age: 30,
    height: 175,
    weight: 75,
    goal: "get_fit",
    style: "strength",
    fitness_level: "beginner",
    days: "Mon, Wed, Fri",
    tod: "evening",
    mins: 30,
  });

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/profile`);
        const p = await res.json();
        const next = {
          name: p.name ?? "Athlete",
          gender: p.gender ?? "unspecified",
          age: p.age ?? 30,
          height: p.height_cm ?? 175,
          weight: p.weight_kg ?? 75,
          goal: p.goal ?? "get_fit",
          style: p.preferred_style ?? "strength",
          fitness_level: p.fitness_level ?? "beginner",
          days: toNiceDaysString(p?.schedule?.days),
          tod: p?.schedule?.time_of_day ?? "evening",
          mins: Number(p?.schedule?.session_minutes ?? 30),
        };
        setForm(next);
      } catch {
        // keep defaults
      } finally {
        setLoadingProfile(false);
      }
    })();
  }, []);

  const { supported: voiceOK, active: voiceActive, listening, interim, startWizard, stopWizard } =
    useVoiceWizard(form, (f) => setForm(f));

  async function saveProfile() {
    setSaving(true);
    try {
      const minutes = Number(form.mins);
      const payload = {
        name: form.name,
        gender: form.gender,
        age: Number(form.age),
        height_cm: Number(form.height),
        weight_kg: Number(form.weight),
        goal: form.goal,
        preferred_style: form.style,
        fitness_level: form.fitness_level,
        schedule: {
          days: toDay3Array(form.days),
          time_of_day: form.tod,
          session_minutes: Number.isFinite(minutes) ? minutes : 30,
        },
      };
      const res = await fetch(`${API_BASE}/api/profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Save failed: ${res.status}`);
      setSaved(true);
      navigate("/dashboard");
    } catch (e) {
      alert(e.message || "Failed to save profile.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <div className="container hero">
        <h1>
          Hi! I’ll plan your week and track calories from{" "}
          <span className="accent">squats</span> & <span className="accent">push-ups</span>.
        </h1>
        <p>Fill the form <b>or</b> use the voice onboarding below.</p>

        {/* Profile form */}
        <div className="card">
          <div className="card-title">Your basics</div>

          {loadingProfile ? (
            <div className="dim">Loading profile…</div>
          ) : (
            <div className="form-grid form-grid-3">
              <div className="form-col" style={{ gridColumn: "1 / span 2" }}>
                <label>Name</label>
                <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
{/* Gender */}
<div>
  <label>Gender</label>
  <select
    value={form.gender}
    onChange={(e) => setForm({ ...form, gender: e.target.value })}
  >
    <option value="unspecified">Rather not say</option>
    <option value="female">Female</option>
    <option value="male">Male</option>
    <option value="nonbinary">Non-binary</option>
  </select>
</div>

              <div>
                <label>Goal</label>
                <select
                    value={form.goal}
                    onChange={(e) => setForm({ ...form, goal: e.target.value })}
                >
                    <option value="lose_weight">Lose weight</option>
                    <option value="get_fit">Get fit</option>
                    <option value="gain_muscle">Gain muscle</option>
                    <option value="get_stronger">Get stronger</option>
                    <option value="more_flexible">More flexible</option>
                </select>
              </div>

              <div>
                <label>Age (years)</label>
                <input
                  type="number"
                  min={10}
                  max={90}
                  value={form.age}
                  onChange={(e) => setForm({ ...form, age: e.target.value })}
                />
              </div>

              <div>
                <label>Height (cm)</label>
                <input
                  type="number"
                  min={120}
                  max={230}
                  value={form.height}
                  onChange={(e) => setForm({ ...form, height: e.target.value })}
                />
              </div>

              <div>
                <label>Weight (kg)</label>
                <input
                  type="number"
                  min={30}
                  max={250}
                  value={form.weight}
                  onChange={(e) => setForm({ ...form, weight: e.target.value })}
                />
              </div>

              <div>
                <label>Preferred style</label>
                <select value={form.style} onChange={(e) => setForm({ ...form, style: e.target.value })}>
                  <option value="cardio">Cardio</option>
                  <option value="hiit">HIIT</option>
                  <option value="yoga">Yoga</option>
                  <option value="strength">Strength</option>
                </select>
              </div>

              <div>
                <label>Fitness level</label>
                <select
                  value={form.fitness_level}
                  onChange={(e) => setForm({ ...form, fitness_level: e.target.value })}
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="expert">Expert</option>
                </select>
              </div>

              <div>
                <label>Training days</label>
                <input
                  value={form.days}
                  onChange={(e) => {
                    const dedup = toNiceDaysString(toDay3Array(e.target.value));
                    setForm({ ...form, days: dedup });
                  }}
                  placeholder="Mon, Wed, Fri"
                />
                <div className="help mt-8">Tip: comma or space separated</div>
              </div>

              <div>
                <label>Time of day</label>
                <select value={form.tod} onChange={(e) => setForm({ ...form, tod: e.target.value })}>
                  <option value="morning">Morning</option>
                  <option value="afternoon">Afternoon</option>
                  <option value="evening">Evening</option>
                </select>
              </div>

              <div>
                <label>Session length (min)</label>
                <input
                  type="number"
                  min={10}
                  max={120}
                  value={form.mins}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setForm({ ...form, mins: Number.isFinite(v) ? v : "" });
                  }}
                />
              </div>
            </div>
          )}

          <div className="btn-row mt-16" style={{ alignItems: "center" }}>
            <button
              className={cls("btn", "primary")}
              onClick={saveProfile}
              disabled={saving || loadingProfile}
              style={{ opacity: saving ? 0.7 : 1 }}
            >
              {saving ? "Saving…" : "Save profile"}
            </button>

            <span className={cls("chip", saved ? "" : "dim")}>{saved ? "Profile saved." : " "}</span>
          </div>
        </div>

        {/* Voice Onboarding */}
        <div className="card mt-16">
          <div className="card-title">Voice onboarding</div>
          {!voiceOK ? (
            <div className="dim">
              Your browser does not support Web Speech. Use Chrome/Edge on desktop and allow the microphone.
            </div>
          ) : (
            <>
              <p className="mb-12">
                I’ll ask: <i>name → gender → age → height → weight → goal → style → fitness → days → time → minutes</i>.
              </p>
              <div className="btn-row">
                {!voiceActive ? (
                  <button className="btn primary" onClick={startWizard} disabled={loadingProfile}>
                    🎙️ Start voice setup
                  </button>
                ) : (
                  <button className="btn danger" onClick={stopWizard}>
                    ⏹ Stop
                  </button>
                )}
                <span className="chip">{voiceActive ? (listening ? "Listening…" : "Speaking…") : "Idle"}</span>
              </div>
              {voiceActive && (
                <div className="help mt-8">
                  {listening ? `Say your answer… ${interim ? ` (${interim})` : ""}` : "Listening paused while speaking…"}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------
   Dashboard (Generate sessions + meal plan)
-------------------------------------------------*/
function Dashboard() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  const [allergy, setAllergy] = useState("");
  const [generating, setGenerating] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [mealPlan, setMealPlan] = useState([]);
  const [running, setRunning] = useState(false);
  const [lastDashboard, setLastDashboard] = useState(null);


  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/profile`);
        const data = await res.json();
        setProfile(data);
      } catch {}
      setLoading(false);
    })();
  }, []);

  async function handleGenerate() {
    if (!profile) return;
    setGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allergy: allergy || "" }),
      });
      if (!res.ok) throw new Error("Failed to generate plan.");
      const data = await res.json();
      setSessions(data.sessions || []);
      setMealPlan(data.meals || []);
    } catch (e) {
      alert(e.message || "Failed to generate.");
    } finally {
      setGenerating(false);
    }
  }
  async function startFirstSessionFromCard(card) {
    if (!card || !Array.isArray(card.exercises)) {
      alert("This session has no exercises.");
      return;
    }
    // Find Squats & Push-ups on the card (case/spacing tolerant)
    const squat = findExercise(card, "squats") || findExercise(card, "squat");
    const push  = findExercise(card, "pushups") || findExercise(card, "pushup") || findExercise(card, "push");
  
    if (!squat || !push) {
      alert("First session must include Squats and Push-ups.");
      return;
    }
  
    const squat_sets = Number(squat.sets);
    const squat_reps = Number(squat.reps);
    const push_sets  = Number(push.sets);
    const push_reps  = Number(push.reps);
  
    if (![squat_sets, squat_reps, push_sets, push_reps].every(Number.isFinite)) {
      alert("Missing or invalid sets/reps on the card.");
      return;
    }
  
    setRunning(true);
    try {
      const res = await fetch(`${API_BASE}/api/run-first-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ squat_sets, squat_reps, push_sets, push_reps }),
      });
      if (!res.ok) throw new Error(`Run failed: ${res.status}`);
      const dashboard = await res.json();
      setLastDashboard(dashboard);
      // optional toast
      alert("Session complete. Dashboard summary saved.");
    } catch (e) {
      alert(e.message || "Could not start session.");
    } finally {
      setRunning(false);
    }
  }
  
  return (
    <div>
      <div className="container">
        <div className="card session-highlight">
          <div>
            <div className="card-title">Your dashboard</div>
            <div className="dim">See plan, profile, and quick actions</div>
          </div>
        <Link className="btn ghost" to="/">Edit profile</Link>
        </div>


        {lastDashboard && (
  <div className="chip" style={{ marginTop: 8 }}>
    Last session: Squat {lastDashboard?.quick?.squat_label || "—"} • Push {lastDashboard?.quick?.push_label || "—"}
  </div>
)}



        <div className="grid-2 mt-16">
          {/* Profile */}
          <div className="card">
            <div className="card-title">Profile</div>
            {loading ? (
              <div className="dim">Loading…</div>
            ) : profile ? (
              <div className="profile-card">
                <div className="profile-card__header">
                  <div className="profile-card__avatar">
                    {(profile.name || "A")[0]}
                  </div>
                  <div>
                    <div className="profile-card__name">{profile.name}</div>
                    <div className="profile-card__subtitle">
                      {profile.preferred_style} • {profile.schedule?.time_of_day}
                    </div>
                  </div>
                </div>
                <div className="profile-card__grid">
                  <div className="row"><span className="label">Gender</span><span className="value">{profile.gender}</span></div>
                  <div className="row"><span className="label">Age</span><span className="value">{profile.age}</span></div>
                  <div className="row"><span className="label">Height</span><span className="value">{profile.height_cm} cm</span></div>
                  <div className="row"><span className="label">Weight</span><span className="value">{profile.weight_kg} kg</span></div>
                  <div className="row"><span className="label">Goal</span><span className="value">{(profile.goal || "").replace("_"," ")}</span></div>
                  <div className="row"><span className="label">Style</span><span className="value badge">{profile.preferred_style}</span></div>
                  <div className="row"><span className="label">Fitness</span><span className="value">{profile.fitness_level || "—"}</span></div>
                  <div className="row row--full">
                    <span className="label">Schedule</span>
                    <span className="value">
                      {(profile.schedule?.days || []).map((d) => day3ToNice[d] || d).join(", ")} • {profile.schedule?.session_minutes} min
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="dim">No profile found.</div>
            )}
          </div>

          {/* Generate plan + allergy input */}
          <div className="card">
            <div className="card-title">Generate plan & meals</div>
            
            <div className="mb-8">
              <label>Food allergy / avoid (optional)</label>
              <input
                placeholder="e.g., peanuts, gluten, dairy"
                value={allergy}
                onChange={(e) => setAllergy(e.target.value)}
              />
              <div className="help mt-8">This will be considered for the 7-day meal ideas.</div>
            </div>
            <div className="help mb-8">
              Rules: if your preferred style is <b>cardio</b>, sessions must include both <b>Squats</b> and <b>Push-ups</b>. The first cardio session should include only those two.
            </div>
            
            <div className="btn-row">
              <button className="btn primary" onClick={handleGenerate} disabled={!profile || generating}>
                {generating ? "Generating…" : "Generate plan for me"}
              </button>
              {!profile && <span className="chip dim">Save a profile first</span>}
            </div>
            
          </div>

          <Link className="btn primary" to="/showcase" style={{ textDecoration: 'none' }}>
  Open LIVE Showcase
</Link>


        </div>

        {/* This week's sessions */}
        <div className="card mt-16">
          <div className="card-title">This week’s sessions</div>
          {!sessions.length ? (
            <div className="dim">No sessions yet. Click “Generate plan for me”.</div>
          ) : (
            <div className="grid-3">
              {sessions.map((s, i) => {
                const duration = Number(s.duration ?? s.minutes ?? 0);
                const goalCals = Math.round(
                  s.target_calories ?? s.targetCalories ?? (duration ? duration * 7 : 0)
                );
                const cpm = duration > 0 && goalCals > 0 ? Math.round(goalCals / duration) : null;

                return (
                  <div className="plan-tile" key={i}>
                    <div className="title">
                      {iconForStyle(s.type)} &nbsp;
                      {String(s.day).toUpperCase()} • {s.type} • {duration} min
                    </div>

                    <div className="meta mt-8" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                      <span className="chip">Intensity: {s.intensity}</span>
                      <span className="chip">
                        Goal: {goalCals} kcal{cpm ? ` (${cpm} kcal/min)` : ""}
                      </span>
                    </div>

                    <div className="hr" />

                    <div className="meta">
                      <span className="label" style={{ opacity: 0.9 }}>Exercises:</span>
                      <div className="exercise-badges">
                        {Array.isArray(s.exercises) && s.exercises.length ? (
                          s.exercises.map((e, idx) => (
                            <ExerciseBadge key={idx} name={`${e.name} ${e.sets}×${e.reps}`} />
                          ))
                        ) : (
                          <span className="dim">—</span>
                        )}
                      </div>
                    </div>

                    <div className="btn-row mt-12">
  {i === 0 ? (
    <button
      className="btn small primary"
      onClick={() => startFirstSessionFromCard(s)}
      disabled={running}
      title="Runs Squats first, then Push-ups, using sets/reps from this card"
    >
      {running ? "Running…" : "Start"}
    </button>
  ) : (
    <button className="btn small ghost" disabled title="Start is only on the first session">
      Start
    </button>
  )}
  <button className="btn small ghost">Swap day</button>
</div>

                  </div>
                );
              })}
            </div>
          )}
        </div>

        
        {/* Meal prep calendar */}
        <div className="card mt-16">
          <div className="card-title">Meal prep calendar (this week)</div>
          

          {!mealPlan.length ? (
            <div className="dim">No meals yet. Click “Generate plan for me”.</div>
            
          ) : (
            <div
              className="meal-calendar"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 16,
                marginTop: 12,
              }}
            >
              {orderDays.map((k) => {
                const entry =
                  mealPlan.find((d) => (d?.day ? three(String(d.day)) : "") === k) || null;
                const label = day3ToNice[k] || k.toUpperCase();

                return (
                  <div
                    key={k}
                    className="plan-tile"
                    style={{
                      padding: 14,
                      display: "flex",
                      flexDirection: "column",
                      borderRadius: 14,
                      overflow: "visible",
                    }}
                  >
                    <div
                      className="title"
                      style={{
                        fontWeight: 700,
                        letterSpacing: 0.2,
                        marginBottom: 8,
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {label}
                    </div>

                    {!entry || !Array.isArray(entry.meals) || !entry.meals.length ? (
                      <div className="dim" style={{ marginTop: 6 }}>—</div>
                    ) : (
                      <ul
                        className="mt-8"
                        style={{
                          marginTop: 6,
                          paddingLeft: 0,
                          listStyleType: "disc",
                          listStylePosition: "inside",
                          display: "flex",
                          flexDirection: "column",
                          gap: 6,
                          lineHeight: 1.45,
                          overflowWrap: "anywhere",
                          wordBreak: "break-word",
                        }}
                      >
                        {entry.meals.map((m, j) => (
                          <li key={j} className="meta" style={{ opacity: 0.95 }}>
                            {m}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

/* ------------------------------------------------
   App & Mount
-------------------------------------------------*/
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/showcase" element={<ShowcaseDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

const root = createRoot(document.getElementById("root"));
root.render(<App />);
