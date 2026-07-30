import { EMOTION_EMOJI } from "../utils/emotion"

const riskColor = (level) =>
  level === "HIGH"
    ? "text-red-400"
    : level === "MEDIUM"
    ? "text-amber-400"
    : "text-green-400"

const SessionTimeline = ({ timeline = [] }) => {

  return (

    <div className="bg-gray-900 rounded-xl p-6">

      <h2 className="text-2xl text-cyan-400 mb-4">
        Timeline
      </h2>

      <div className="overflow-auto max-h-[500px]">

        {timeline.length === 0 ? (

          <p className="text-gray-500">
            Waiting for session…
          </p>

        ) : (

          <ul className="space-y-2">

            {[...timeline].reverse().map((entry, index) => (

              <li
                key={index}
                className="flex items-center gap-3 border-b border-gray-800 pb-1"
              >

                <span className="font-mono text-gray-400">
                  {entry.time}
                </span>

                {entry.type === "emotion" ? (

                  <span className="text-yellow-400 font-semibold">
                    {EMOTION_EMOJI[entry.label] ?? ""}
                    {" "}
                    {entry.label}
                  </span>

                ) : (

                  <span
                    className={
                      "font-semibold " + riskColor(entry.label)
                    }
                  >
                    → Risk: {entry.label}
                    {entry.score != null && ` (${entry.score})`}
                  </span>

                )}

              </li>

            ))}

          </ul>

        )}

      </div>

    </div>
  )
}

export default SessionTimeline
