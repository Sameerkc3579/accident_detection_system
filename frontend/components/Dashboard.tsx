import React from 'react';

interface DashboardProps {
    originalVideoUrl: string;
    processedVideoUrl: string | null;
    status: string | null;
    confidence: number;
}

export default function Dashboard({ originalVideoUrl, processedVideoUrl, status, confidence }: DashboardProps) {
    // Determine style based on status
    const isAccident = status === "Accident Detected";
    const isSafe = status === "Normal Traffic";

    let statusColor = "text-slate-100";
    if (isAccident) statusColor = "text-red-500 animate-pulse text-neon-red";
    if (isSafe) statusColor = "text-neon-blue";

    return (
        <div className="space-y-6 animate-in slide-in-from-bottom-5 duration-700">
            {/* Command Bar */}
            <div className="glass-panel rounded-none border-l-4 border-l-red-500 p-6 flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex items-center gap-6">
                    <div className={`w-4 h-4 rounded-sm ${isAccident ? 'bg-red-500 animate-ping' : 'bg-emerald-500'}`} />
                    <div>
                        <p className="text-xs text-slate-500 font-mono tracking-widest uppercase mb-1">System Status</p>
                        <h2 className={`text-3xl font-black uppercase tracking-tight ${statusColor}`}>
                            {status || "STANDBY..."}
                        </h2>
                    </div>
                </div>

                <div className="flex gap-12 font-mono">
                    <div className="text-right">
                        <p className="text-xs text-slate-500 uppercase">Analysis Confidence</p>
                        <div className="text-2xl font-bold text-white flex items-baseline justify-end gap-1">
                            {(confidence * 100).toFixed(1)}
                            <span className="text-sm text-slate-600">%</span>
                        </div>
                    </div>
                    <div className="text-right hidden md:block">
                        <p className="text-xs text-slate-500 uppercase">Session ID</p>
                        <p className="text-2xl font-bold text-slate-400">#8X-29</p>
                    </div>
                </div>
            </div>

            {/* Video Feeds */}
            <div className="grid md:grid-cols-2 gap-6 h-[400px]">
                {/* Original Feed */}
                <div className="glass-panel p-1 flex flex-col h-full relative group">
                    <div className="absolute top-4 left-4 z-10 bg-black/50 px-2 py-1 border border-slate-700">
                        <span className="text-[10px] font-mono text-slate-300 tracking-widest">FEED_A // RAW</span>
                    </div>

                    {/* Corners */}
                    <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-slate-500" />
                    <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-slate-500" />
                    <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-slate-500" />
                    <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-slate-500" />

                    <div className="flex-1 bg-black overflow-hidden relative scanline">
                        <video
                            src={originalVideoUrl}
                            controls
                            className="w-full h-full object-contain opacity-80 group-hover:opacity-100 transition-opacity"
                        />
                    </div>
                </div>

                {/* Processed Feed */}
                <div className={`glass-panel p-1 flex flex-col h-full relative transition-all duration-500 ${isAccident ? 'border-red-500/50 shadow-[0_0_50px_rgba(220,38,38,0.2)]' : ''}`}>
                    <div className="absolute top-4 left-4 z-10 bg-red-900/40 px-2 py-1 border border-red-500/30 backdrop-blur-sm">
                        <span className="text-[10px] font-mono text-red-400 tracking-widest flex items-center gap-2">
                            FEED_B // AI_ANALYSIS
                            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
                        </span>
                    </div>

                    <div className="flex-1 bg-black/90 overflow-hidden relative flex items-center justify-center">
                        {processedVideoUrl ? (
                            <video
                                src={processedVideoUrl}
                                controls
                                autoPlay
                                loop
                                muted
                                className="w-full h-full object-contain"
                            />
                        ) : (
                            <div className="text-center space-y-4">
                                <div className="relative w-16 h-16 mx-auto">
                                    <div className="absolute inset-0 border-t-2 border-red-500 rounded-full animate-spin" />
                                    <div className="absolute inset-2 border-b-2 border-red-500/50 rounded-full animate-spin direction-reverse" />
                                </div>
                                <p className="text-red-500 font-mono text-sm animate-pulse tracking-widest">PROCESSING STREAM...</p>
                            </div>
                        )}

                        {/* Overlay HUD */}
                        <div className="absolute inset-0 pointer-events-none border border-white/5 m-4">
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 opacity-20">
                                <div className="w-full h-[1px] bg-red-500 absolute top-1/2" />
                                <div className="h-full w-[1px] bg-red-500 absolute left-1/2" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
