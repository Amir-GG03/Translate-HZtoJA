import { useState, useEffect } from 'react';
import { Mic, MicOff, Languages, ArrowRightLeft, Loader2, Copy, Check, BrainCircuit } from 'lucide-react';
import useSpeechToText from './useSpeechToText';

// URL de tu API
const API_URL = "http://127.0.0.1:8000/translate";

interface AttentionData {
  matrix: number[][];
  tokens: string[];
}

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [attentionData, setAttentionData] = useState<AttentionData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Hook de audio
  const { isListening, transcript, startListening, stopListening, hasSupport } = useSpeechToText('es-ES');

  useEffect(() => {
    if (transcript) setInputText(transcript);
  }, [transcript]);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setOutputText('');
    setAttentionData(null); // Limpiar atención anterior

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          source_lang: "spa_Latn",
          target_lang: "ita_Latn"
        })
      });

      if (!response.ok) throw new Error("Error en la red");
      
      const data = await response.json();
      setOutputText(data.translation);
      
      // Guardar datos de atención si la API los manda
      if (data.attention) {
        setAttentionData(data.attention);
      }

    } catch (error) {
      console.error("Error:", error);
      setOutputText("Error: No se pudo conectar con el servidor de IA.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="min-h-screen flex flex-col items-center p-4 md:p-8 bg-slate-50">
      
      {/* Header */}
      <div className="mb-8 text-center mt-4">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="p-3 bg-blue-600 rounded-full text-white shadow-lg shadow-blue-200">
            <Languages size={28} />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-slate-800 tracking-tight">Traductor NLLB</h1>
        </div>
        <p className="text-slate-500 font-medium">Fine-Tuning con LoRA • Español ↔ Italiano</p>
      </div>

      {/* Main Translation Card */}
      <div className="bg-white w-full max-w-5xl rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden mb-8">
        <div className="flex flex-col md:flex-row h-auto md:h-[400px]">
          
          {/* Panel Izquierdo: Input */}
          <div className="flex-1 p-6 border-b md:border-b-0 md:border-r border-slate-100 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <span className="font-bold text-slate-700 flex items-center gap-2 bg-slate-100 px-3 py-1 rounded-full text-sm">
                🇪🇸 Español
              </span>
              {hasSupport && (
                <button 
                  onClick={isListening ? stopListening : startListening}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    isListening 
                      ? 'bg-red-100 text-red-600 hover:bg-red-200 animate-pulse' 
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {isListening ? <><MicOff size={14}/> Escuchando...</> : <><Mic size={14}/> Dictar</>}
                </button>
              )}
            </div>
            
            <textarea
              className="flex-1 w-full bg-transparent border-none resize-none outline-none text-xl text-slate-800 placeholder:text-slate-300"
              placeholder="Escribe algo aquí..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleTranslate(); }}}
            />
          </div>

          {/* Botón Central */}
           <div className="relative flex items-center justify-center bg-slate-50 p-2 md:w-16">
              <div className="absolute inset-0 md:w-[1px] md:h-full w-full h-[1px] bg-slate-100 m-auto"></div>
              <button 
                onClick={handleTranslate} 
                disabled={isLoading || !inputText.trim()}
                className="z-10 bg-blue-600 text-white p-4 rounded-2xl shadow-lg shadow-blue-200 hover:scale-105 hover:bg-blue-700 transition-all disabled:opacity-50 disabled:scale-100 active:scale-95"
              >
                {isLoading ? <Loader2 className="animate-spin"/> : <ArrowRightLeft />}
              </button>
           </div>

          {/* Panel Derecho: Output */}
          <div className="flex-1 p-6 bg-blue-50/30 flex flex-col">
            <div className="flex justify-between items-center mb-4">
               <span className="font-bold text-blue-800 flex items-center gap-2 bg-blue-100 px-3 py-1 rounded-full text-sm">
                🇮🇹 Italiano
              </span>
              {outputText && (
                  <button onClick={handleCopy} className="text-slate-400 hover:text-blue-600 transition-colors">
                      {copied ? <Check size={18}/> : <Copy size={18}/>}
                  </button>
              )}
            </div>

            <div className="flex-1 overflow-auto flex items-start">
                {isLoading ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 gap-3">
                        <Loader2 size={30} className="animate-spin text-blue-400"/>
                        <span className="text-sm font-medium animate-pulse">Procesando tensores...</span>
                    </div>
                ) : outputText ? (
                    <p className="text-2xl font-medium text-slate-800 leading-relaxed">{outputText}</p>
                ) : (
                    <p className="text-slate-300 text-xl font-light">La traducción aparecerá aquí.</p>
                )}
            </div>
          </div>
        </div>
      </div>

      {/* Sección de Visualización de Atención */}
      {attentionData && !isLoading && (
        <div className="w-full max-w-5xl animate-fade-in-up">
          <div className="flex items-center gap-2 mb-4 text-slate-700">
            <BrainCircuit className="text-purple-600" />
            <h2 className="text-xl font-bold">Matriz de Atención del Encoder</h2>
          </div>
          
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-100 overflow-x-auto">
            <p className="text-sm text-slate-500 mb-6 max-w-2xl">
              Este mapa muestra cómo el modelo relaciona las palabras entre sí para entender el contexto. 
              Los cuadros <span className="font-bold text-blue-600">más oscuros</span> indican una relación gramatical o semántica más fuerte.
            </p>
            
            <AttentionHeatmap matrix={attentionData.matrix} tokens={attentionData.tokens} />
          </div>
        </div>
      )}

    </div>
  );
}

// --- SUB-COMPONENTE: Mapa de Calor ---
const AttentionHeatmap = ({ matrix, tokens }: { matrix: number[][], tokens: string[] }) => {
  // Truco: Si la frase es muy larga, limitamos para que no rompa la UI
  const displayLimit = 20; 
  const limitedTokens = tokens.slice(0, displayLimit);
  const limitedMatrix = matrix.slice(0, displayLimit).map(row => row.slice(0, displayLimit));

  return (
    <div className="inline-block min-w-full">
      <div 
        className="grid gap-1"
        style={{ 
          gridTemplateColumns: `auto repeat(${limitedTokens.length}, minmax(40px, 1fr))` 
        }}
      >
        {/* Cabecera vacía (esquina) */}
        <div className="h-8"></div>

        {/* Eje X (Tokens Superiores) */}
        {limitedTokens.map((token, i) => (
          <div key={`head-${i}`} className="text-xs font-mono text-slate-500 -rotate-45 origin-bottom-left translate-x-4 mb-2 truncate">
            {token}
          </div>
        ))}

        {/* Filas de la matriz */}
        {limitedMatrix.map((row, i) => (
          <>
            {/* Eje Y (Tokens Izquierda) */}
            <div key={`row-label-${i}`} className="text-xs font-mono text-slate-500 flex items-center justify-end pr-3">
              {limitedTokens[i]}
            </div>

            {/* Celdas */}
            {row.map((value, j) => (
              <div
                key={`cell-${i}-${j}`}
                className="aspect-square rounded-sm transition-all hover:scale-125 hover:z-10 hover:ring-2 ring-purple-400 relative group cursor-crosshair"
                style={{
                  backgroundColor: `rgba(79, 70, 229, ${value})`, // Color Indigo base con opacidad dinámica
                  opacity: Math.max(0.1, value * 3) // Boost visual para que se vean mejor los valores bajos
                }}
              >
                {/* Tooltip simple al pasar el mouse */}
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-[10px] rounded hidden group-hover:block whitespace-nowrap z-20 pointer-events-none">
                  {value.toFixed(4)}
                </div>
              </div>
            ))}
          </>
        ))}
      </div>
      {tokens.length > displayLimit && (
        <p className="text-xs text-slate-400 mt-4 text-center italic">
          * Visualización truncada a los primeros {displayLimit} tokens por espacio.
        </p>
      )}
    </div>
  );
};

export default App;