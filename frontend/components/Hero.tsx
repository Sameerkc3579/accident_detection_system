import React, { useEffect, useState } from 'react';

export default function Hero() {
  const [text, setText] = useState('');
  const fullText = "AI_POWERED_TRAFFIC_ANALYSIS_SYSTEM";

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setText(fullText.slice(0, index));
      index++;
      if (index > fullText.length) clearInterval(interval);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative py-20 md:py-32 flex flex-col items-center justify-center overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20 bg-[url('/grid.svg')] z-0"></div>

      <div className="z-10 text-center space-y-6 max-w-4xl mx-auto px-4">
        <div className="inline-block mb-4 animate-pulse">
          <span className="px-3 py-1 border border-red-500 text-red-500 text-xs font-mono tracking-widest uppercase bg-red-500/10 rounded-sm">
            System Online
          </span>
        </div>

        <h1 className="text-5xl md:text-9xl font-black text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-500 drop-shadow-[0_0_20px_rgba(255,255,255,0.3)] animate-glitch relative break-words w-full">
          SENTINEL
          <span className="absolute -inset-1 text-red-500 opacity-20 blur-lg animate-pulse hidden md:block">SENTINEL</span>
        </h1>

        <div className="h-auto min-h-[2rem] font-mono text-neon-blue tracking-normal md:tracking-widest text-[10px] md:text-xl break-all md:break-words">
          {text}<span className="animate-blink">_</span>
        </div>

        <p className="text-slate-400 font-light text-sm md:text-lg max-w-2xl mx-auto border-l-2 border-red-500 pl-4 md:pl-6 text-left">
          Advanced computer vision protocols deployed for real-time hazard detection.
          Monitor video feeds with autonomous accident recognition algorithms.
        </p>
      </div>

      {/* Decorative lines */}
      <div className="absolute bottom-0 w-full h-[1px] bg-gradient-to-r from-transparent via-red-500/50 to-transparent"></div>
    </div>
  );
}
