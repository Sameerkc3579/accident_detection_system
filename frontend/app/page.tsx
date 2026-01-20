"use client";

import React, { useState, useEffect, useRef } from 'react';
import Hero from '@/components/Hero';
import UploadZone from '@/components/UploadZone';
import Dashboard from '@/components/Dashboard';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{ status: string; confidence: number } | null>(null);
  const [clientId, setClientId] = useState<string>("");
  const socketRef = useRef<WebSocket | null>(null);

  // Generate unique Client ID and connect to WebSocket on mount
  useEffect(() => {
    const id = Math.random().toString(36).substring(7);
    setClientId(id);

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000/ws";
    const socket = new WebSocket(`${wsUrl}/${id}`);

    socket.onopen = () => {
      console.log("Connected to WebSocket");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("WebSocket Message:", data);

      if (data.type === "accident_alert") {
        setResult({
          status: data.status,
          confidence: data.confidence // Use confidence from alert
        });
        // We do NOT stop processing here; we just show the alert
      } else if (data.type === "complete") {
        setIsProcessing(false);
        setProcessedUrl(data.video_url);
        setResult({
          status: data.status,
          confidence: data.confidence
        });
      } else if (data.type === "error") {
        setIsProcessing(false);
        alert(`Error: ${data.message}`);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket Error:", error);
    };

    socketRef.current = socket;

    return () => {
      if (socket.readyState === 1) { // OPEN
        socket.close();
      }
    };
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setOriginalUrl(url);
    // Reset previous results
    setProcessedUrl(null);
    setResult(null);
  };

  const startDetection = async () => {
    if (!file || !clientId) return;

    setIsProcessing(true);
    setResult(null); // Reset result on new start

    // Create FormData
    const formData = new FormData();
    formData.append("file", file);
    formData.append("client_id", clientId);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Detection request failed");
      }

      // We don't wait for 'data' here as the real result comes via WebSocket
      console.log("Detection started via API");

    } catch (error) {
      console.error("Error:", error);
      let errorMessage = "System Offline or Error Processing Video";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      const targetUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      alert(`Connection Failed! \n\nTarget URL: ${targetUrl}\n\nError: ${errorMessage}`);
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen relative selection:bg-red-500/30 overflow-x-hidden">
      {/* Top Status Bar */}
      <div className="fixed top-0 left-0 w-full z-50 bg-black/80 backdrop-blur-md border-b border-white/10 px-4 py-2 flex justify-between items-center text-[10px] md:text-xs font-mono text-slate-400">
        <div className="flex gap-4">
          <span className="text-emerald-500">● SYSTEM_ONLINE</span>
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
                  className="btn-primary"
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
                status={result?.status || (isProcessing ? "PROCESSING STREAM..." : null)}
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
