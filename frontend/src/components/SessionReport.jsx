import { EMOTION_EMOJI } from "../utils/emotion"

const levelColor = (level) =>
  level === "HIGH"
    ? "text-red-400"
    : level === "MEDIUM"
    ? "text-amber-400"
    : "text-green-400"

const Stat = ({ label, value, sub, color = "text-white" }) => (
  <div className="bg-gray-800 rounded-lg p-4">
    <p className="text-gray-400 text-sm">{label}</p>
    <p className={"text-2xl font-bold " + color}>{value}</p>
    {sub && <p className="text-gray-500 text-xs mt-1">{sub}</p>}
  </div>
)

const Distribution = ({ distribution = {} }) => {
  const total =
    Object.values(distribution).reduce((a, b) => a + b, 0) || 1

  const color = {
    LOW: "bg-green-500",
    MEDIUM: "bg-amber-400",
    HIGH: "bg-red-500",
  }

  return (
    <div className="space-y-2">
      {["LOW", "MEDIUM", "HIGH"].map((level) => {
        const count = distribution[level] ?? 0
        const pct = Math.round((count / total) * 100)
        return (
          <div key={level} className="flex items-center gap-3">
            <span className="w-16 text-sm text-gray-400">{level}</span>
            <div className="flex-1 h-3 bg-gray-700 rounded">
              <div
                className={"h-3 rounded " + (color[level] ?? "bg-gray-500")}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-12 text-right text-sm text-gray-400">
              {pct}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

const SessionReport = ({ report }) => {
  if (!report) {
    return (
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-2xl text-cyan-400 mb-2">Interview Report</h2>
        <p className="text-gray-500">
          Generate a report after recording a session.
        </p>
      </div>
    )
  }

  if (!report.frame_count) {
    return (
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-2xl text-cyan-400 mb-2">Interview Report</h2>
        <p className="text-gray-500">{report.summary}</p>
      </div>
    )
  }

  const { stress, emotion, risk, blink, gaze } = report

  return (
    <div className="bg-gray-900 rounded-xl p-6 space-y-6">

      <div>
        <h2 className="text-2xl text-cyan-400 mb-1">Interview Report</h2>
        <p className="text-gray-500 text-sm">
          {report.session_id} · {report.duration} · {report.frame_count} frames
        </p>
      </div>

      <p className="text-gray-200 leading-relaxed">{report.summary}</p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat
          label="Avg Stress"
          value={`${stress.average_percent}%`}
          sub={stress.average_level}
          color={levelColor(stress.average_level)}
        />
        <Stat
          label="Highest Stress"
          value={stress.highest_level}
          sub={`at ${stress.highest_at}`}
          color={levelColor(stress.highest_level)}
        />
        <Stat
          label="Avg Risk"
          value={`${risk.average_score}`}
          sub="/ 100"
          color={levelColor(risk.dominant_level)}
        />
        <Stat
          label="Peak Risk"
          value={`${risk.highest_score}`}
          sub={`at ${risk.highest_at}`}
          color={levelColor(risk.highest_score >= 70 ? "HIGH" : risk.highest_score >= 40 ? "MEDIUM" : "LOW")}
        />
        <Stat
          label="Dominant Emotion"
          value={`${EMOTION_EMOJI[emotion.dominant] ?? ""} ${emotion.dominant}`}
          sub={`${emotion.stability_percent}% stable`}
          color="text-yellow-400"
        />
        <Stat
          label="Blink Rate"
          value={`${blink.per_minute}/min`}
          sub={`${blink.total} total`}
          color="text-pink-400"
        />
        <Stat
          label="Gaze Stability"
          value={`${gaze.average_stability_percent}%`}
          sub={`${gaze.avoidance_percent}% aversion`}
          color="text-cyan-400"
        />
        <Stat
          label="Final Deception"
          value={risk.final_deception}
          color={levelColor(risk.final_deception)}
        />
      </div>

      <div>
        <h3 className="text-lg text-gray-300 mb-3">Stress Distribution</h3>
        <Distribution distribution={stress.distribution} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        <div>
          <h3 className="text-lg text-gray-300 mb-3">Emotion Timeline</h3>
          <ul className="space-y-1">
            {emotion.timeline.map((entry, index) => (
              <li key={index} className="flex items-center gap-3">
                <span className="font-mono text-gray-500 w-12">
                  {entry.time}
                </span>
                <span className="text-yellow-400">
                  {EMOTION_EMOJI[entry.emotion] ?? ""} {entry.emotion}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="text-lg text-gray-300 mb-3">Risk Timeline</h3>
          <ul className="space-y-1">
            {risk.timeline.map((entry, index) => (
              <li key={index} className="flex items-center gap-3">
                <span className="font-mono text-gray-500 w-12">
                  {entry.time}
                </span>
                <span className={"font-semibold " + levelColor(entry.level)}>
                  {entry.level} ({entry.score})
                </span>
              </li>
            ))}
          </ul>
        </div>

      </div>

      <div>
        <h3 className="text-lg text-gray-300 mb-3">Recommendations</h3>
        <ul className="space-y-2">
          {report.recommendations.map((rec, index) => (
            <li
              key={index}
              className="flex gap-2 text-gray-200 bg-gray-800 rounded-lg p-3"
            >
              <span className="text-cyan-400">▸</span>
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

    </div>
  )
}

export default SessionReport
