import React, {
  useEffect,
  useState
} from "react"

import {
  getSessionSummary,
  getSessionData
} from "./services/api"

import LiveCamera from "./components/LiveCamera"

const App = () => {

  const [summary, setSummary] = useState(null)

  const [data, setData] = useState([])

  const [liveAI, setLiveAI] = useState(null)

  useEffect(() => {

    loadData()

  }, [])

  const loadData = async () => {

    try {

      const summaryData =
        await getSessionSummary()

      const sessionData =
        await getSessionData()

      setSummary(summaryData)

      setData(sessionData)

    } catch (error) {

      console.error(error)
    }
  }

  return (

    <div className="min-h-screen bg-black text-white p-8">

      <h1 className="text-4xl font-bold mb-8 text-cyan-400">
        🧠 MaCoDeR Live Dashboard
      </h1>

      <div className="grid grid-cols-2 gap-6 mb-8">

        <LiveCamera
          onAIUpdate={(aiData) => {

            setLiveAI(aiData)

            setData(prev => [
              aiData,
              ...prev
            ])

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

      {summary && (

        <div className="grid grid-cols-3 gap-6 mb-10">

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Records
            </h2>

            <p className="text-3xl font-bold text-green-400">
              {summary.records}
            </p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Emotion Confidence
            </h2>

            <p className="text-3xl font-bold text-yellow-400">
              {summary.avg_emotion_confidence.toFixed(1)}%
            </p>
          </div>

          <div className="bg-gray-900 p-6 rounded-xl">
            <h2 className="text-xl mb-2">
              Temporal Confidence
            </h2>

            <p className="text-3xl font-bold text-cyan-400">
              {summary.avg_temporal_confidence.toFixed(1)}%
            </p>
          </div>

        </div>
      )}

      <div className="bg-gray-900 rounded-xl p-6">

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