import { useState, useEffect, useCallback } from 'react';

// Definición de tipos para la API del navegador (que TypeScript a veces no conoce por defecto)
interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start: () => void;
    stop: () => void;
    onresult: (event: any) => void;
    onerror: (event: any) => void;
    onend: () => void;
}

declare global {
    interface Window {
        SpeechRecognition: any;
        webkitSpeechRecognition: any;
    }
}

const useSpeechToText = (lang: string = 'es-ES') => {
    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [recognition, setRecognition] = useState<SpeechRecognition | null>(null);

    useEffect(() => {
        // Verificar si el navegador soporta la API
        const SpeechRecognitionFunc = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRecognitionFunc) {
            const recognitionInstance = new SpeechRecognitionFunc();
            recognitionInstance.continuous = true; // Seguir escuchando aunque haya pausas
            recognitionInstance.interimResults = true; // Mostrar resultados mientras hablas
            recognitionInstance.lang = lang; // Idioma español

            recognitionInstance.onresult = (event: any) => {
                let currentTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    currentTranscript += event.results[i][0].transcript;
                }
                setTranscript(currentTranscript);
            };

            recognitionInstance.onerror = (event: any) => {
                console.error("Error de reconocimiento de voz:", event.error);
                setIsListening(false);
            };

            recognitionInstance.onend = () => {
                setIsListening(false);
            };

            setRecognition(recognitionInstance);
        }
    }, [lang]);

    const startListening = useCallback(() => {
        if (recognition && !isListening) {
            try {
                setTranscript(''); // Limpiar texto anterior
                recognition.start();
                setIsListening(true);
            } catch (error) {
                console.error("No se pudo iniciar el micrófono", error)
            }
        }
    }, [recognition, isListening]);

    const stopListening = useCallback(() => {
        if (recognition && isListening) {
            recognition.stop();
            setIsListening(false);
        }
    }, [recognition, isListening]);

    return { isListening, transcript, startListening, stopListening, hasSupport: !!recognition };
};

export default useSpeechToText;