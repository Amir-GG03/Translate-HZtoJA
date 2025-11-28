import { useState, useEffect } from 'react';
import { Mic, MicOff, Languages, ArrowRightLeft, Loader2, Copy, Check } from 'lucide-react';
import useSpeechToText from './useSpeechToText';

// URL de tu API FastAPI local (asegúrate de que uvicorn esté corriendo)
const API_URL = "http://127.0.0.1:8000/translate";

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Usamos nuestro hook de audio
  const { isListening, transcript, startListening, stopListening, hasSupport } = useSpeechToText('es-ES');

  // Actualizamos el input text cuando el micrófono detecta algo
  useEffect(() => {
    if (transcript) {
      // Si estamos escuchando, reemplazamos. Si paramos, podríamos añadir.
      // Para este ejemplo, el dictado reemplaza lo que había.
      setInputText(transcript);
    }
  }, [transcript]);


  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setOutputText(''); // Limpiar anterior

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          source_lang: "spa_Latn", // Fijo en Español
          target_lang: "ita_Latn"  // Fijo en Italiano
        })
      });

      if (!response.ok) throw new Error("Error en la red");
      const data = await response.json();
      setOutputText(data.translation);

    } catch (error) {
      console.error("Error traduciendo:", error);
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

  // Componente de botón reutilizable con estilos de Tailwind
  const Button = ({ onClick, disabled, children, variant = 'primary', className = '' }: any) => {
    const baseStyle = "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";
    const variants = {
      primary: "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500",
      secondary: "bg-slate-200 text-slate-800 hover:bg-slate-300 focus:ring-slate-400",
      danger: "bg-red-500 text-white hover:bg-red-600 focus:ring-red-500 animate-pulse",
    };
    return <button onClick={onClick} disabled={disabled} className={`${baseStyle} ${variants[variant as keyof typeof variants]} ${className}`}>{children}</button>
  };


  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">

      {/* Header */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="p-3 bg-blue-100 rounded-full text-blue-600">
            <Languages size={32} />
          </div>
          <h1 className="text-4xl font-bold text-slate-800">Traductor IA</h1>
        </div>
        <p className="text-slate-600">Español coloquial a Italiano natural</p>
      </div>

      {/* Main Card */}
      <div className="bg-white w-full max-w-4xl rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div className="flex flex-col md:flex-row">

          {/* Panel Izquierdo: Input (Español) */}
          <div className="flex-1 p-6 border-b md:border-b-0 md:border-r border-slate-100">
            <div className="flex justify-between items-center mb-4">
              <span className="font-semibold text-slate-700 flex items-center gap-2">
                🇪🇸 Español (Origen)
              </span>

              {hasSupport && (
                <Button
                  onClick={isListening ? stopListening : startListening}
                  variant={isListening ? 'danger' : 'secondary'}
                  className="text-sm py-1.5"
                >
                  {isListening ? <><MicOff size={16} /> Detener</> : <><Mic size={16} /> Dictar</>}
                </Button>
              )}
            </div>

            <div className="relative">
              <textarea
                className="w-full h-48 p-4 bg-slate-50 rounded-xl border-none resize-none focus:ring-2 focus:ring-blue-500 outline-none text-lg text-slate-800 placeholder:text-slate-400 transition-all"
                placeholder="Escribe algo o usa el micrófono..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleTranslate(); } }}
              />
              {isListening && <span className="absolute bottom-4 right-4 flex h-3 w-3"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span><span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span></span>}
            </div>

          </div>

          {/* Botón Central de Acción */}
          <div className="flex items-center justify-center p-4 bg-slate-50 md:bg-white">
            <Button
              onClick={handleTranslate}
              disabled={isLoading || !inputText.trim()}
              className="md:rotate-0 rotate-90 rounded-full p-4"
            >
              {isLoading ? <Loader2 className="animate-spin" /> : <ArrowRightLeft />}
            </Button>
          </div>

          {/* Panel Derecho: Output (Italiano) */}
          <div className="flex-1 p-6 bg-blue-50/50">
            <div className="flex justify-between items-center mb-4">
              <span className="font-semibold text-slate-700 flex items-center gap-2">
                🇮🇹 Italiano (Destino)
              </span>
              {outputText && (
                <button onClick={handleCopy} className="text-slate-500 hover:text-blue-600 transition-colors">
                  {copied ? <Check size={20} className="text-green-500" /> : <Copy size={20} />}
                </button>
              )}
            </div>

            <div className="w-full h-48 p-4 bg-white/80 rounded-xl border border-blue-100 overflow-auto text-lg text-slate-800 relative flex items-start">
              {isLoading ? (
                <div className="flex items-center gap-2 text-slate-400 italic">
                  <Loader2 size={20} className="animate-spin" /> Traduciendo...
                </div>
              ) : outputText ? (
                <p>{outputText}</p>
              ) : (
                <p className="text-slate-400 italic">La traducción aparecerá aquí.</p>
              )}
            </div>
          </div>
        </div>
      </div>
      <p className="text-slate-400 text-sm mt-6">
        Modelo: NLLB-200 (600M) + Adaptador LoRA personalizado. Corriendo localmente en Apple M2.
      </p>
    </div>
  );
}

export default App;