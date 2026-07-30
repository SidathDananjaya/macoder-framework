import React, {
  useEffect,
  useRef,
  useState
} from "react"

import {
  exportSessionCsv,
  exportSessionJson,
  getSessionReport,
  exportSessionReport,
  getSessionInterpretation
} from "./services/api"

import LiveCamera from "./components/LiveCamera"

import SessionTimeline from "./components/SessionTimeline"

import LiveCharts from "./components/LiveCharts"

import SessionReport from "./components/SessionReport"

import SessionInterpretation from "./components/SessionInterpretation"

const fmt = (ms) => {
  const s = Math.floor(ms / 1000)
  const mm = String(Math.floor(s / 60)).padStart(2, "0")
  const ss = String(s % 60).padStart(2, "0")
  return `${mm}:${ss}`
}

const App = () => {

  const [data, setData] = useState([])

  const [liveAI, setLiveAI] = useState(null)

  const [timeline, setTimeline] = useState([])

  const [history, setHistory] = useState([])

  const [report, setReport] = useState(null)

  const [interpretation, setInterpretation] = useState(null)
  const [interpreting, setInterpreting] = useState(false)

  const [recording, setRecording] = useState(false)

  const tlRef = useRef({
    startMs: null,
    lastEmotion: null,
    lastRisk: null
  })

  const startSession = () => {
    tlRef.current = {
      startMs: null,
      lastEmotion: null,
      lastRisk: null
    }
    setTimeline([])
    setHistory([])
    setData([])
    setReport(null)
    setInterpretation(null)
  }

  const pushHistory = (aiData) => {
    setHistory(prev => {
      const point = {
        t: prev.length,
        stress: Math.round((aiData.fused_stress_score ?? 0) * 100),
        risk: aiData.risk_score ?? 0,
        movement: Math.round(aiData.movement_score ?? 0),
        confidence: Math.round((aiData.fusion_confidence ?? 0) * 100),
        blinks: aiData.blink_count ?? 0,
        emotion: aiData.emotion ?? "neutral"
      }
      return [...prev, point].slice(-60)
    })
  }

  const pushTimeline = (aiData) => {

    const tl = tlRef.current

    if (tl.startMs === null) {
      tl.startMs = Date.now()
    }

    const time = fmt(Date.now() - tl.startMs)

    const entries = []

    if (aiData.emotion && aiData.emotion !== tl.lastEmotion) {
      entries.push({
        time,
        type: "emotion",
        label: aiData.emotion
      })
      tl.lastEmotion = aiData.emotion
    }

    if (aiData.risk_level && aiData.risk_level !== tl.lastRisk) {
      entries.push({
        time,
        type: "risk",
        label: aiData.risk_level,
        score: aiData.risk_score
      })
      tl.lastRisk = aiData.risk_level
    }

    if (entries.length > 0) {
      setTimeline(prev => [...prev, ...entries].slice(-200))
    }
  }

  const avg = (fn) =>
    data.length
      ? data.reduce((sum, row) => sum + (fn(row) ?? 0), 0) / data.length
      : 0

  const sessionStats = {
    records: data.length,
    emotionConfidence:
      avg(row => row.visual_emotion_confidence ?? row.cognitive_confidence) * 100,
    temporalConfidence:
      avg(row => row.fusion_confidence) * 100
  }

  return (

    <div className="min-h-screen bg-black text-white p-8">

      <h1 className="text-4xl font-bold mb-6 text-cyan-400">
        🧠 MaCoDeR Live Dashboard
      </h1>

      <div className="flex items-center gap-4 mb-8">

        {!recording ? (
          <button
            className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-semibold"
            onClick={() => setRecording(true)}
          >
            ▶ Start Session
          </button>
        ) : (
          <button
            className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-semibold"
            onClick={() => setRecording(false)}
          >
            ■ Stop Session
          </button>
        )}

        <span className="text-gray-400">
          {recording
            ? "Recording - camera & microphone active."
            : "Stopped - start a session, or generate a report."}
        </span>

      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">

        <LiveCamera
          active={recording}
          onSessionStart={startSession}
          onAIUpdate={(aiData) => {

            setLiveAI(aiData)

            setData(prev => [
              aiData,
              ...prev
            ])

            pushTimeline(aiData)

            pushHistory(aiData)

          }}
        />

        <div className="bg-gray-900 rounded-xl p-6">

          <h2 className="text-2xl text-cyan-400 mb-4">
            Live AI Status
          </h2>

          {liveAI && (

            <div className="space-y-4">

              <div>
                <p className="text-gray-400">
                  Emotion
                </p>

                <p className="text-3xl font-bold text-yellow-400">
                  {liveAI.emotion}
                </p>
              </div>

              <div>
                <p className="text-gray-400">
                  Stress Level
                </p>

                <p className="text-3xl font-bold text-red-400">
                  {liveAI.stress_level}
                </p>
              </div>

              <div>
                <p className="text-gray-400">
                  Risk Score
                </p>

                <p className="text-3xl font-bold text-white">
                  {liveAI.risk_score ?? 0}
                  <span className="text-lg text-gray-400">
                    {" "}
                    / 100
                  </span>
                  <span className="text-lg ml-2 text-gray-300">
                    ({liveAI.risk_level ?? "—"})
                  </span>
                </p>

                <div className="w-full h-2 bg-gray-700 rounded mt-2">
                  <div
                    className={
                      "h-2 rounded " +
                      ((liveAI.risk_score ?? 0) >= 70
                        ? "bg-red-500"
                        : (liveAI.risk_score ?? 0) >= 40
                        ? "bg-amber-400"
                        : "bg-green-500")
                    }
                    style={{
                      width: `${Math.min(liveAI.risk_score ?? 0, 100)}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <p className="text-gray-400">
                  Confidence
                </p>

                <p className="text-3xl font-bold text-cyan-400">
                  {liveAI.fusion_confidence}
                </p>
              </div>

              <div>
                <p className="text-gray-400">
                  Blink Count
                </p>

                <p className="text-3xl font-bold text-pink-400">
                  {liveAI.blink_count}
                </p>
              </div>

            </div>
          )}

        </div>

      </div>

      {data.length > 0 && (

        <div className="grid grid-cols-3 gap-6 mb-10">

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Records
            </h2>

            <p className="text-3xl font-bold text-green-400">
              {sessionStats.records}
            </p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Emotion Confidence
            </h2>

            <p className="text-3xl font-bold text-yellow-400">
              {sessionStats.emotionConfidence.toFixed(1)}%
            </p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Temporal Confidence
            </h2>

            <p className="text-3xl font-bold text-cyan-400">
              {sessionStats.temporalConfidence.toFixed(1)}%
            </p>
          </div>

        </div>
      )}

      <div className="mb-10">
        <LiveCharts history={history} />
      </div>

      <div className="mb-10">
        <SessionTimeline timeline={timeline} />
      </div>

      <div className="mb-10">
        <SessionReport report={report} />
      </div>

      <div className="mb-10">
        <SessionInterpretation
          interpretation={interpretation}
          loading={interpreting}
        />
      </div>

      <div className="bg-gray-900 rounded-xl p-6">

        <div className="flex gap-4 mb-6">

          <button
            className="bg-cyan-600 hover:bg-cyan-700 px-5 py-3 rounded-lg"
            onClick={async () => {
              const res = await exportSessionCsv()
              alert(
                res.saved
                  ? `Session CSV saved:\n${res.file}`
                  : "Nothing recorded yet."
              )
            }}
          >
            Export CSV
          </button>

          <button
            className="bg-emerald-600 hover:bg-emerald-700 px-5 py-3 rounded-lg"
            onClick={async () => {
              const res = await exportSessionJson()
              alert(
                res.saved
                  ? `Session recording saved:\n${res.file}`
                  : "Nothing recorded yet."
              )
            }}
          >
            Save Recording (JSON)
          </button>

          <button
            className="bg-indigo-600 hover:bg-indigo-700 px-5 py-3 rounded-lg"
            onClick={async () => {
              const res = await getSessionReport()
              setReport(res)
            }}
          >
            Generate Report
          </button>

          <button
            className="bg-purple-600 hover:bg-purple-700 px-5 py-3 rounded-lg"
            onClick={async () => {
              const res = await exportSessionReport()
              alert(
                res.saved
                  ? `Report saved:\n${res.file}`
                  : "Nothing recorded yet."
              )
            }}
          >
            Save Report
          </button>

          <button
            className="bg-fuchsia-600 hover:bg-fuchsia-700 px-5 py-3 rounded-lg disabled:opacity-50"
            disabled={interpreting}
            onClick={async () => {
              setInterpreting(true)
              try {
                const res = await getSessionInterpretation()
                setInterpretation(res)
              } finally {
                setInterpreting(false)
              }
            }}
          >
            AI Interpretation
          </button>

        </div>

        <h2 className="text-2xl mb-4">
          Live Session Data
        </h2>

        <div className="overflow-auto max-h-[500px]">

          <table className="w-full">

            <thead>

              <tr className="text-left border-b border-gray-700">

                <th className="p-2">Emotion</th>
                <th className="p-2">Temporal</th>
                <th className="p-2">Stress</th>
                <th className="p-2">Confidence</th>

              </tr>

            </thead>

            <tbody>

              {data.map((row, index) => (

                <tr
                  key={index}
                  className="border-b border-gray-800"
                >

                  <td className="p-2">
                    {row.emotion}
                  </td>

                  <td className="p-2">
                    {row.temporal_emotion}
                  </td>

                  <td className="p-2">
                    {row.stress_level}
                  </td>

                  <td className="p-2">
                    {row.fusion_confidence}
                  </td>

                </tr>

              ))}

            </tbody>

          </table>

        </div>

      </div>

    </div>
  )
}

export default App
