// UI HINT: "Level Theme" Card Component
const GlassCard = ({ children, className }) => (
  <div className={`
    relative overflow-hidden
    bg-slate-900/60 backdrop-blur-xl
    border border-white/10
    rounded-3xl
    shadow-2xl shadow-black/50
    p-6
    ${className}
  `}>
    {/* Gradient Blob for "Glow" effect */}
    <div className="absolute -top-20 -right-20 w-60 h-60 bg-blue-600/20 rounded-full blur-3xl pointer-events-none" />
    
    <div className="relative z-10">
      {children}
    </div>
  </div>
);

// UI HINT: Gradient Text
const GradientText = ({ text }) => (
  <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400 font-bold">
    {text}
  </span>
);