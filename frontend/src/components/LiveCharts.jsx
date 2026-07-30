import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"

import { EMOTION_EMOJI } from "../utils/emotion"

const METRICS = [
  { key: "stress", label: "Stress", color: "#d95926", domain: [0, 100], unit: "" },
  { key: "risk", label: "Risk", color: "#e66767", domain: [0, 100], unit: "" },
  { key: "movement", label: "Movement", color: "#9085e9", domain: [0, "auto"], unit: "" },
  { key: "confidence", label: "Confidence", color: "#3987e5", domain: [0, 100], unit: "%" },
  { key: "blinks", label: "Blinks", color: "#199e70", domain: [0, "auto"], unit: "" },
]

const AXIS = "#898781"
const SURFACE = "#1a1a19"

const Spark = ({ metric, history }) => {

  const latest =
    history.length > 0
      ? history[history.length - 1][metric.key]
      : 0

  return (

    <div className="bg-gray-900 rounded-xl p-4">

      <div className="flex items-baseline justify-between mb-1">
        <p className="text-sm text-gray-400">
          {metric.label}
        </p>
        <p
          className="text-lg font-bold"
          style={{ color: metric.color }}
        >
          {latest}
          {metric.unit}
        </p>
      </div>

      <ResponsiveContainer width="100%" height={70}>

        <AreaChart
          data={history}
          margin={{ top: 4, right: 2, bottom: 0, left: 2 }}
        >

          <defs>
            <linearGradient
              id={`fill-${metric.key}`}
              x1="0" y1="0" x2="0" y2="1"
            >
              <stop offset="0%" stopColor={metric.color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={metric.color} stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <XAxis dataKey="t" hide />

          <YAxis hide domain={metric.domain} />

          <Tooltip
            contentStyle={{
              background: SURFACE,
              border: "1px solid rgba(255,255,255,0.10)",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            labelStyle={{ color: AXIS }}
            cursor={{ stroke: AXIS, strokeWidth: 1 }}
          />

          <Area
            type="monotone"
            dataKey={metric.key}
            stroke={metric.color}
            strokeWidth={2}
            fill={`url(#fill-${metric.key})`}
            dot={false}
            isAnimationActive={false}
          />

        </AreaChart>

      </ResponsiveContainer>

    </div>
  )
}

const LiveCharts = ({ history = [] }) => {

  const recentEmotions = history.slice(-24)

  return (

    <div className="bg-gray-900 rounded-xl p-6">

      <h2 className="text-2xl text-cyan-400 mb-4">
        Live Charts
      </h2>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">

        {METRICS.map((metric) => (
          <Spark
            key={metric.key}
            metric={metric}
            history={history}
          />
        ))}

        <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
          <p className="text-sm text-gray-400 mb-2">
            Emotion
          </p>

          {recentEmotions.length === 0 ? (
            <p className="text-gray-600 text-sm">
              Waiting…
            </p>
          ) : (
            <div className="flex flex-wrap gap-1 text-xl leading-7">
              {recentEmotions.map((point, index) => (
                <span
                  key={index}
                  title={point.emotion}
                >
                  {EMOTION_EMOJI[point.emotion] ?? "•"}
                </span>
              ))}
            </div>
          )}
        </div>

      </div>

    </div>
  )
}

export default LiveCharts
