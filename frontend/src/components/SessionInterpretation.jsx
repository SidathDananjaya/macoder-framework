const SessionInterpretation = ({ interpretation, loading }) => {

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-2xl text-cyan-400 mb-2">AI Interpretation</h2>
        <p className="text-gray-400 animate-pulse">
          Analysing the session with the local model… (this can take a minute)
        </p>
      </div>
    )
  }

  if (!interpretation) {
    return (
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-2xl text-cyan-400 mb-2">AI Interpretation</h2>
        <p className="text-gray-500">
          Generate an AI interpretation after recording a session.
        </p>
      </div>
    )
  }

  const {
    llm_available,
    model,
    headline,
    observations = [],
    overall_assessment,
    confidence_note
  } = interpretation

  return (
    <div className="bg-gray-900 rounded-xl p-6 space-y-5">

      <div className="flex items-center justify-between gap-4">
        <h2 className="text-2xl text-cyan-400">AI Interpretation</h2>
        <span
          className={
            "text-xs px-2 py-1 rounded " +
            (llm_available
              ? "bg-emerald-900 text-emerald-300"
              : "bg-amber-900 text-amber-300")
          }
        >
          {llm_available ? `LLM · ${model}` : "Rule-based fallback"}
        </span>
      </div>

      {!llm_available && (
        <p className="text-amber-300 text-sm bg-amber-950/40 border border-amber-800 rounded-lg p-3">
          The local language model was unavailable, so a rule-based summary is
          shown. Make sure <code className="text-amber-200">ollama serve</code> is
          running and the model is pulled to enable the AI interpretation.
        </p>
      )}

      <p className="text-xl text-gray-100 font-semibold leading-snug">
        {headline}
      </p>

      {observations.length > 0 && (
        <div>
          <h3 className="text-lg text-gray-300 mb-2">Observations</h3>
          <ul className="space-y-2">
            {observations.map((obs, index) => (
              <li key={index} className="flex gap-2 text-gray-200">
                <span className="text-cyan-400">▸</span>
                <span>{obs}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {overall_assessment && (
        <div>
          <h3 className="text-lg text-gray-300 mb-2">Overall Assessment</h3>
          <p className="text-gray-200 leading-relaxed">
            {overall_assessment}
          </p>
        </div>
      )}

      {confidence_note && (
        <p className="text-gray-400 text-sm italic border-t border-gray-800 pt-4">
          {confidence_note}
        </p>
      )}

    </div>
  )
}

export default SessionInterpretation
