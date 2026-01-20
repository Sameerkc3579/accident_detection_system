"use client";

import React, { useState, useRef } from 'react';

interface UploadZoneProps {
    onFileSelected: (file: File) => void;
    isProcessing: boolean;
}

export default function UploadZone({ onFileSelected, isProcessing }: UploadZoneProps) {
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelected(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelected(e.target.files[0]);
        }
    };

    return (
        <form
            id="upload"
            onDragEnter={handleDrag}
            onSubmit={(e) => e.preventDefault()}
            className="max-w-3xl mx-auto mb-12 relative"
        >
            <input
                ref={inputRef}
                type="file"
                accept="video/mp4,video/webm"
                className="hidden"
                onChange={handleChange}
                disabled={isProcessing}
            />

            {/* Corner Markers */}
            <div className="absolute -top-2 -left-2 w-8 h-8 border-t-2 border-l-2 border-red-500/50" />
            <div className="absolute -top-2 -right-2 w-8 h-8 border-t-2 border-r-2 border-red-500/50" />
            <div className="absolute -bottom-2 -left-2 w-8 h-8 border-b-2 border-l-2 border-red-500/50" />
            <div className="absolute -bottom-2 -right-2 w-8 h-8 border-b-2 border-r-2 border-red-500/50" />

            <div
                className={`
          glass-panel rounded-lg p-12 text-center cursor-pointer transition-all duration-300
          border border-transparent
          ${dragActive ? 'border-neon-red bg-red-900/10' : 'hover:border-red-500/30'}
          ${isProcessing ? 'opacity-50 pointer-events-none grayscale' : ''}
        `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
            >
                <div className="space-y-6 pointer-events-none relative z-10">
                    <div className="relative w-20 h-20 mx-auto flex items-center justify-center">
                        <div className={`absolute inset-0 border border-slate-600 rounded-full ${dragActive ? 'animate-ping border-red-500' : ''}`} />
                        <div className="absolute inset-0 border border-slate-700 rounded-full scale-75" />
                        <svg xmlns="http://www.w3.org/2000/svg" className={`w-8 h-8 ${dragActive ? 'text-red-500' : 'text-slate-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                    </div>

                    <div>
                        <p className="text-xl font-bold text-white tracking-widest uppercase">
                            Initialize Data Stream
                        </p>
                        <p className="text-slate-500 mt-2 font-mono text-xs">
                            [ DROP FOOTAGE OR CLICK TO UPLOAD ]
                        </p>
                    </div>
                </div>
            </div>
        </form>
    );
}
