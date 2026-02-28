export default function Home() {
  return (
    <main className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
      <div className="text-center space-y-4">
        <h1 className="text-6xl font-bold text-[#00f0ff] tracking-tight">
          🜏 AGOTHE OS
        </h1>
        <p className="text-gray-400 text-xl">
          Paulk is online. Systems nominal.
        </p>
        <div className="flex gap-6 justify-center mt-8 text-sm font-mono">
          <span className="text-green-400">● API :8000 ✅</span>
          <span className="text-green-400">● WEB :3000 ✅</span>
          <span className="text-green-400">● BRAIN gemma3:4b ✅</span>
        </div>
        <p className="text-gray-600 text-xs mt-8 font-mono">
          Constraint-Resonance Duality · Dreaming Wake Mode · Agothe Corporation
        </p>
      </div>
    </main>
  )
}
