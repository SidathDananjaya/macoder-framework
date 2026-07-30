import React, {
  useEffect,
  useRef,
  useState
} from "react"

const LiveCamera = ({
  onAIUpdate,
  onSessionStart,
  active
}) => {

  const videoRef = useRef(null)

  const canvasRef = useRef(null)

  const socketRef = useRef(null)

  const streamRef = useRef(null)

  const audioContextRef = useRef(null)

  const processorRef = useRef(null)

  const audioBufferRef = useRef([])

  const [liveAI, setLiveAI] =
    useState(null)

  useEffect(() => {

    if (!active) {
      return
    }

    let stopped = false
    let sendTimer = null

    const captureAndSend = () => {

      if (stopped) return

      const socket = socketRef.current

      if (
        !socket ||
        socket.readyState !== WebSocket.OPEN
      ) return

      if (
        !videoRef.current ||
        !canvasRef.current
      ) return

      const canvas = canvasRef.current

      const ctx = canvas.getContext("2d")

      canvas.width =
        videoRef.current.videoWidth

      canvas.height =
        videoRef.current.videoHeight

      ctx.drawImage(
        videoRef.current,
        0,
        0
      )

      const frame = canvas.toDataURL(
        "image/jpeg",
        0.7
      )

      const audioChunk = audioBufferRef.current
      audioBufferRef.current = []

      socket.send(
        JSON.stringify({
          frame: frame,
          audio: audioChunk,
          sample_rate:
            audioContextRef.current
              ? audioContextRef.current.sampleRate
              : 22050
        })
      )
    }

    const startCamera = async () => {

      try {

        const stream =
          await navigator.mediaDevices
            .getUserMedia({
              video: true,
              audio: true
            })

        streamRef.current = stream

        const AudioContext =
          window.AudioContext ||
          window.webkitAudioContext

        audioContextRef.current =
          new AudioContext()

        const source =
          audioContextRef.current
            .createMediaStreamSource(
              stream
            )

        processorRef.current =
          audioContextRef.current
            .createScriptProcessor(
              4096,
              1,
              1
            )

        source.connect(
          processorRef.current
        )

        processorRef.current.connect(
          audioContextRef.current.destination
        )

        processorRef.current.onaudioprocess =
          (event) => {

            const channelData =
              event.inputBuffer
                .getChannelData(0)

            const buffer = audioBufferRef.current

            for (let i = 0; i < channelData.length; i++) {
              buffer.push(channelData[i])
            }
          }

        if (videoRef.current) {

          videoRef.current.srcObject =
            stream
        }

        const socket = new WebSocket(
          "ws://127.0.0.1:8000/ws/live"
        )

        socketRef.current = socket

        socket.onopen = () => {

          console.log(
            "WebSocket Connected"
          )

          if (onSessionStart) {
            onSessionStart()
          }

          captureAndSend()
        }

        socket.onmessage = (event) => {

          const data =
            JSON.parse(event.data)

          setLiveAI(data)

          if (onAIUpdate) {
            onAIUpdate(data)
          }

          if (!stopped) {
            sendTimer = setTimeout(captureAndSend, 100)
          }
        }

        socket.onerror = (err) => {

          console.error(
            "WebSocket Error:",
            err
          )
        }

      } catch (err) {

        console.error(
          "Camera Error:",
          err
        )
      }
    }

    startCamera()

    return () => {

      stopped = true

      if (sendTimer) {
        clearTimeout(sendTimer)
      }

      if (socketRef.current) {
        socketRef.current.close()
      }

      if (streamRef.current) {

        streamRef.current
          .getTracks()
          .forEach(track =>
            track.stop()
          )
      }

      if (processorRef.current) {
        processorRef.current.disconnect()
      }

      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
    }

  }, [active])

  return (

    <div className="relative bg-gray-900 rounded-xl overflow-hidden">

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full rounded-xl"
      />

      <canvas
        ref={canvasRef}
        className="hidden"
      />

      {liveAI?.quality_warning && (

        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-20 bg-amber-500/90 text-black font-bold px-4 py-2 rounded-lg shadow-lg">
          ⚠ Poor signal quality. Results may be less reliable.
        </div>
      )}

      {liveAI && (

        <div className="absolute top-4 left-4 bg-black/70 p-4 rounded-xl space-y-2">

          <p className="text-cyan-400 font-bold">
            Emotion:
            {" "}
            {liveAI.emotion}
          </p>

          <p className="text-yellow-400 font-bold">
            Stress:
            {" "}
            {liveAI.stress_level}
          </p>

          <p className="text-green-400 font-bold">
            Audio Emotion:
            {" "}
            {liveAI.audio_emotion}
          </p>

          <p className="text-red-400 font-bold">
            Audio Stress:
            {" "}
            {liveAI.audio_stress}
          </p>

          <p className="text-pink-400 font-bold">
            Blinks:
            {" "}
            {liveAI.blink_count}
          </p>

          <p className="text-sky-300 font-bold">
            Video Quality:
            {" "}
            {Math.round((liveAI.video_quality ?? 0) * 100)}
            %
          </p>

          <p className="text-teal-300 font-bold">
            Audio Quality:
            {" "}
            {Math.round((liveAI.audio_quality ?? 0) * 100)}
            %
          </p>

          <hr className="border-gray-600" />

          <div>
            <p className="text-white font-bold">
              Risk Score:
              {" "}
              {liveAI.risk_score ?? 0}
              {" / 100 "}
              <span className="text-gray-300">
                ({liveAI.risk_level ?? "—"})
              </span>
            </p>

            <div className="w-40 h-2 bg-gray-700 rounded mt-1">
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

          <p className="text-orange-400 font-bold">
            Fusion Score:
            {" "}
            {liveAI.fusion_score}
          </p>

          <p className="text-purple-400 font-bold">
            Cognitive Load:
            {" "}
            {liveAI.cognitive_load}
          </p>

          <p className="text-rose-400 font-bold">
            Deception Risk:
            {" "}
            {liveAI.deception_risk}
          </p>

        </div>
      )}

      {liveAI?.session_memory?.["30s"] && (

        <div className="absolute top-4 right-4 bg-black/70 p-4 rounded-xl space-y-1 text-sm">

          <p className="text-cyan-300 font-bold mb-1">
            Session Memory (30s)
          </p>

          <p className="text-gray-200">
            Dominant Emotion:
            {" "}
            {liveAI.session_memory["30s"].dominant_emotion}
          </p>

          <p className="text-gray-200">
            Stress Trend:
            {" "}
            {(() => {
              const t =
                liveAI.session_memory["30s"].stress_trend
              return t === "INCREASING"
                ? "▲ INCREASING"
                : t === "DECREASING"
                ? "▼ DECREASING"
                : "— STABLE"
            })()}
          </p>

          <p className="text-gray-200">
            Blink Freq:
            {" "}
            {liveAI.session_memory["30s"].blink_frequency}
            {" "}
            /s
          </p>

          <p className="text-gray-200">
            Head Movement:
            {" "}
            {liveAI.session_memory["30s"].avg_head_movement}
          </p>

          <p className="text-gray-200">
            Speech Intensity:
            {" "}
            {liveAI.session_memory["30s"].avg_speech_intensity}
          </p>

          <p className="text-gray-200">
            Emotion Changes:
            {" "}
            {liveAI.session_memory["30s"].emotion_changes}
          </p>

          {liveAI.session_memory.current_vs_historical && (

            <p className="text-yellow-300 font-bold pt-1">
              Now vs History:
              {" "}
              {liveAI.session_memory
                .current_vs_historical.direction}
              {" "}
              ({liveAI.session_memory
                .current_vs_historical.delta})
            </p>
          )}

        </div>
      )}

    </div>
  )
}

export default LiveCamera
