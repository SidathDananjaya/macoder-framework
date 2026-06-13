import React, {
  useEffect,
  useRef,
  useState
} from "react"

const LiveCamera = ({
  onAIUpdate
}) => {

  const videoRef = useRef(null)

  const canvasRef = useRef(null)

  const socketRef = useRef(null)

  const streamRef = useRef(null)

  const [liveAI, setLiveAI] =
    useState(null)

  useEffect(() => {

    let frameInterval = null

    const startCamera = async () => {

      try {

        const stream =
          await navigator.mediaDevices
            .getUserMedia({
              video: true,
              audio: false
            })

        streamRef.current = stream

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

          frameInterval = setInterval(() => {

            if (
              !videoRef.current ||
              !canvasRef.current
            ) return

            const canvas =
              canvasRef.current

            const ctx =
              canvas.getContext("2d")

            canvas.width =
              videoRef.current.videoWidth

            canvas.height =
              videoRef.current.videoHeight

            ctx.drawImage(
              videoRef.current,
              0,
              0
            )

            const frame =
              canvas.toDataURL(
                "image/jpeg",
                0.7
              )

            socket.send(frame)

          }, 100)

        }

        socket.onmessage = (event) => {

          const data =
            JSON.parse(event.data)

          setLiveAI(data)

          if (onAIUpdate) {
            onAIUpdate(data)
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

      if (frameInterval) {
        clearInterval(frameInterval)
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
    }

  }, [])

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

          <p className="text-pink-400 font-bold">
            Blinks:
            {" "}
            {liveAI.blink_count}
          </p>

        </div>
      )}

    </div>
  )
}

export default LiveCamera