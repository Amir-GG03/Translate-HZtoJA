import { useState, useEffect } from 'react';
import { Mic, MicOff, Languages, ArrowRightLeft, Loader2, Copy, Check, BrainCircuit } from 'lucide-react';
import useSpeechToText from './useSpeechToText';

const API_URL = "http://127.0.0.1:8000/translate";

interface AttentionData {
  matrix: number[][];
  src_tokens: string[];
  tgt_tokens: string[];
}

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [attentionData, setAttentionData] = useState<AttentionData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const { isListening, transcript, startListening, stopListening, hasSupport } = useSpeechToText('es-ES');

  useEffect(() => {
    if (transcript) setInputText(transcript);
  }, [transcript]);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setOutputText('');
    setAttentionData(null);

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
      
      if (data.attention) {
        setAttentionData(data.attention);
      }

    } catch (error) {
      console.error("Error:", error);
      setOutputText("Error al conectar con la API.");
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
      
      <div className="mb-8 text-center mt-4">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="p-3 bg-blue-600 rounded-full text-white shadow-lg shadow-blue-200">
            <Languages size={28} />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-slate-800 tracking-tight">Traductor NLLB</h1>
        </div>
        <p className="text-slate-500 font-medium">Fine-Tuning Local • Español - Italiano</p>
      </div>

      <div className="bg-white w-full max-w-5xl rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden mb-8">
        <div className="flex flex-col md:flex-row h-auto md:h-[350px]">
          
          {/* Panel Izquierdo: Input */}
          <div className="flex-1 p-6 border-b md:border-b-0 md:border-r border-slate-100 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <span className="font-bold text-slate-700 flex items-center gap-2 bg-slate-100 px-3 py-1 rounded-full text-sm">
                Español
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
                Italiano
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
                        <span className="text-sm font-medium animate-pulse">Inferencia en curso...</span>
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

      {attentionData && !isLoading && (
        <div className="w-full max-w-5xl animate-fade-in-up pb-12">
          <div className="flex items-center gap-2 mb-4 text-slate-700">
            <BrainCircuit className="text-purple-600" />
            <h2 className="text-xl font-bold">Atención Cruzada (Capa 2)</h2>
          </div>
          
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-100 overflow-x-auto">
            <p className="text-sm text-slate-500 mb-6 max-w-2xl">
              Este mapa muestra la <b>Alineación Neuronal</b>. Visualiza qué palabra en español (arriba) utiliza el modelo para generar cada palabra en italiano (izquierda).
            </p>
            
            <AttentionHeatmap 
                matrix={attentionData.matrix} 
                srcTokens={attentionData.src_tokens} 
                tgtTokens={attentionData.tgt_tokens} 
            />
          </div>
        </div>
      )}

    </div>
  );
}

// --- COMPONENTE VISUALIZADOR DE ATENCIÓN CORREGIDO ---
const AttentionHeatmap = ({ matrix, srcTokens, tgtTokens }: { matrix: number[][], srcTokens: string[], tgtTokens: string[] }) => {
  
  // CORRECCIÓN DE ALINEACIÓN:
  // Usamos slice(0, -1). Mantenemos el índice 0 (inicio), solo borramos el índice -1 (final).
  // Esto evita que las columnas se desplacen a la izquierda.
  
  const cleanSrcTokens = srcTokens.slice(0, -1);
  const cleanTgtTokens = tgtTokens.slice(0, -1);
  
  // Recortamos la matriz igual: filas 0 a -1, columnas 0 a -1
  const cleanMatrix = matrix
    .slice(0, -1) 
    .map(row => row.slice(0, -1));

  const displayLimit = 28;
  const displaySrc = cleanSrcTokens.slice(0, displayLimit);
  const displayTgt = cleanTgtTokens.slice(0, displayLimit);
  const displayMatrix = cleanMatrix.slice(0, displayLimit).map(row => row.slice(0, displayLimit));

  return (
    <div className="inline-block min-w-full overflow-x-auto p-2">
      <div 
        className="grid gap-[1px]"
        style={{ 
          gridTemplateColumns: `min-content repeat(${displaySrc.length}, minmax(45px, 1fr))` 
        }}
      >
        {/* Esquina superior izquierda vacía */}
        <div className="h-24"></div>

        {/* Cabeceras Eje X (Español) */}
        {displaySrc.map((token, i) => (
          <div key={`head-${i}`} className="relative h-24 w-full">
             <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-32 -rotate-45 origin-bottom-left text-xs font-mono text-slate-600 font-medium truncate text-left ml-2 hover:text-blue-600 transition-colors cursor-default">
                {token}
             </div>
          </div>
        ))}

        {/* Filas de la matriz */}
        {displayMatrix.map((row, i) => {
            
            // Normalización por fila
            const maxVal = Math.max(...row, 0.00001);
            const minVal = Math.min(...row);

            return (
              <>
                {/* Etiqueta Eje Y (Italiano) */}
                <div key={`row-label-${i}`} className="text-xs font-mono text-slate-600 flex items-center justify-end pr-3 font-bold whitespace-nowrap h-10">
                  {displayTgt[i]}
                </div>

                {/* Celdas de valores */}
                {row.map((val, j) => {
                  
                  let intensity = (val - minVal) / (maxVal - minVal + 0.00001);
                  if (intensity < 0.2) intensity = 0;

                  return (
                    <div
                      key={`cell-${i}-${j}`}
                      className="h-10 w-full rounded-sm border border-slate-50 relative group transition-transform hover:scale-110 hover:z-50"
                      style={{
                        backgroundColor: `rgba(79, 70, 229, ${intensity})`,
                      }}
                    >
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 text-white text-xs rounded shadow-xl hidden group-hover:block z-50 whitespace-nowrap pointer-events-none">
                        <div className="font-bold border-b border-slate-600 pb-1 mb-1 text-center">
                            Peso: {val.toFixed(4)}
                        </div>
                        {displaySrc[j]} ➔ {displayTgt[i]}
                      </div>
                    </div>
                  );
                })}
              </>
            );
        })}
      </div>
    </div>
  );
};

export default App;