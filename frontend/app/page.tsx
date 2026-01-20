"use client";

import React, { useState, useEffect, useRef } from 'react';
import Hero from '@/components/Hero';
import UploadZone from '@/components/UploadZone';
import Dashboard from '@/components/Dashboard';

// ---------------------------------------------------------
// 🔧 CONFIGURATION: CONNECTING TO HUGGING FACE
// ---------------------------------------------------------
// This is your live backend URL from the previous step.
// We treat this as the "Master URL" for the entire app.
const API_BASE_URL = "https://sameerchoudhary67-sentinel-backend.hf.space";
// ---------------------------------------------------------

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{ status: string; confidence: number } | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("Initializing...");
  const [isConnected, setIsConnected] = useState(false);
  const [clientId, setClientId] = useState<string>("");
  const socketRef = useRef<WebSocket | null>(null);

  // Helper to safely derive WS URL
  const getWsUrl = (id: string) => {
    // 1. Get the base URL (Prefer Environment Variable, fallback to Hardcoded Cloud URL)
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || API_BASE_URL;

    // 2. Remove trailing slash if it exists
    const cleanUrl = baseUrl.replace(/\/$/, '');

    // 3. Auto-switch protocol (https -> wss, http -> ws)
    // Hugging Face uses HTTPS, so this will correctly switch to Secure WebSocket (wss://)
    const wsProtocol = cleanUrl.startsWith('https') ? 'wss' : 'ws';
    const wsBase = cleanUrl.replace(/^https?/, wsProtocol);

    return `${wsBase}/ws/${id}`;
  };

  // Generate unique Client ID and connect to WebSocket on mount
  useEffect(() => {
    // Prevent double-connection in React Strict Mode
    if (socketRef.current) return;

    const id = Math.random().toString(36).substring(7);
    setClientId(id);

    const wsUrl = getWsUrl(id);
    console.log("🚀 Connecting to Sentinel Backend:", wsUrl);

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("✅ WebSocket Connected");
      setIsConnected(true);
      setStatusMessage("System Online");
    };

    socket.onclose = (event) => {
      console.log("❌ WebSocket Disconnected", event);
      setIsConnected(false);
      // Only alert if we were actually in the middle of processing
      if (isProcessing) {
        setIsProcessing(false);
        alert("Connection Lost! The server disconnected before finishing.");
      }
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("📩 Message:", data);

      if (data.type === "accident_alert") {
        setResult({
          status: data.status,
          confidence: data.confidence
        });
        // Do NOT stop processing; allow the video to finish
      } else if (data.type === "status_update") {
        setStatusMessage(data.status);
      } else if (data.type === "complete") {
        setIsProcessing(false);
        setProcessedUrl(data.video_url);
        setResult({
          status: data.status,
          confidence: data.confidence
        });
      } else if (data.type === "error") {
        setIsProcessing(false);
        alert(`Server Error: ${data.message}`);
      }
    };

    socket.onerror = (error) => {
      console.error("⚠️ WebSocket Error:", error);
    };

    socketRef.current = socket;

    // Cleanup on unmount
    return () => {
      if (socket.readyState === 1) {
        socket.close();
      }
    };
  }, []); // Empty dependency array = Run once on mount

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setOriginalUrl(url);
    setProcessedUrl(null);
    setResult(null);
  };

  const startDetection = async () => {
    if (!file || !clientId) return;

    setIsProcessing(true);
    setResult(null);
    setStatusMessage("Uploading Video...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("client_id", clientId);

    try {
      // Use the Cloud URL
      const targetUrl = process.env.NEXT_PUBLIC_API_URL || API_BASE_URL;

      const response = await fetch(`${targetUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Detection request failed. Backend might be waking up.");
      }

      console.log("✅ Upload Complete. Waiting for analysis...");

    } catch (error) {
      console.error("❌ Upload Error:", error);
      let errorMessage = "System Offline or Error Processing Video";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      alert(`Connection Failed!\n\nCheck if the Backend is running.\nError: ${errorMessage}`);
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen relative selection:bg-red-500/30 overflow-x-hidden">
      {/* Top Status Bar */}
      <div className="fixed top-0 left-0 w-full z-50 bg-black/80 backdrop-blur-md border-b border-white/10 px-4 py-2 flex justify-between items-center text-[10px] md:text-xs font-mono text-slate-400">
        <div className="flex gap-4">
          <span className={`transition-colors duration-500 ${isConnected ? "text-emerald-500" : "text-red-500"}`}>
            ● {isConnected ? "SYSTEM_ONLINE" : "CONNECTING..."}
          </span>
          <span className="hidden md:inline">NET_SECURE</span>
          <span className="hidden md:inline">LATENCY: 12ms</span>
        </div>
        <div className="flex gap-4">
          <span>V 2.0.4</span>
          <span className="text-red-500">SENTINEL_CORE_ACTIVE</span>
        </div>
      </div>

      <div className="max-w-7xl mx-auto space-y-12 px-4 md:px-6 pt-16 pb-20">
        <Hero />

        {/* Upload Area */}
        {!result && !isProcessing && (
          <div className="space-y-6 animate-in fade-in zoom-in duration-500">
            <UploadZone
              onFileSelected={handleFileSelect}
              isProcessing={isProcessing}
            />

            {file && (
              <div className="text-center animate-fade-in">
                <button
                  onClick={startDetection}
                  className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 shadow-lg shadow-red-900/20 tracking-widest"
                >
                  INITIATE SCAN PROTOCOL
                </button>
              </div>
            )}
          </div>
        )}

        {/* Processing / Results Area */}
        {(isProcessing || result) && (
          <div className="mt-10">
            {originalUrl && (
              <Dashboard
                originalVideoUrl={originalUrl}
                processedVideoUrl={processedUrl}
                status={result?.status || (isProcessing ? statusMessage : null)}
                confidence={result?.confidence || 0}
              />
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 w-full bg-black/80 backdrop-blur-md border-t border-white/10 py-2 text-center text-[10px] text-slate-600 font-mono z-40">
        SENTINEL SECURITY SYSTEMS © 2026 // UNAUTHORIZED ACCESS PROHIBITED
      </footer>
    </main>
  );
}