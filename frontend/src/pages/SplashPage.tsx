import { useState } from 'react';

interface Props {
  onEnter: () => void;
}

const PARTICLE_COUNT = 50;

export function SplashPage({ onEnter }: Props) {
  const [hover, setHover] = useState(false);

  return (
    <div style={{
      width: '100%',
      height: '100vh',
      position: 'relative',
      overflow: 'hidden',
      background: 'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      {/* Animated particles */}
      {Array.from({ length: PARTICLE_COUNT }).map((_, i) => {
        const size = Math.random() * 6 + 3;
        const x = Math.random() * 100;
        const delay = Math.random() * 5;
        const duration = Math.random() * 5 + 4;
        return (
          <div key={i} style={{
            position: 'absolute',
            left: `${x}%`,
            top: '-10px',
            width: size,
            height: size,
            borderRadius: '50%',
            background: `rgba(255, 255, 255, ${Math.random() * 0.4 + 0.1})`,
            animation: `floatDown ${duration}s ${delay}s linear infinite`,
          }} />
        );
      })}

      {/* Decorative rings */}
      <div style={{
        position: 'absolute',
        width: 1000, height: 1000,
        borderRadius: '50%',
        border: '1px solid rgba(255,255,255,0.04)',
        top: '50%', left: '50%',
        transform: 'translate(-50%, -50%)',
      }} />
      <div style={{
        position: 'absolute',
        width: 800, height: 800,
        borderRadius: '50%',
        border: '1px solid rgba(255,255,255,0.06)',
        top: '50%', left: '50%',
        transform: 'translate(-50%, -50%)',
      }} />
      <div style={{
        position: 'absolute',
        width: 600, height: 600,
        borderRadius: '50%',
        border: '1px solid rgba(255,255,255,0.1)',
        top: '50%', left: '50%',
        transform: 'translate(-50%, -50%)',
      }} />

      {/* Gradient glow */}
      <div style={{
        position: 'absolute',
        width: 500, height: 500,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%)',
        top: '25%', left: '25%',
      }} />
      <div style={{
        position: 'absolute',
        width: 400, height: 400,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%)',
        bottom: '15%', right: '20%',
      }} />

      {/* Content */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        textAlign: 'center',
      }}>
        {/* Icon */}
        <div style={{
          fontSize: 100,
          marginBottom: 20,
          animation: 'float 3s ease-in-out infinite',
        }}>
          🤖
        </div>

        {/* Title */}
        <h1 style={{
          margin: 0,
          fontSize: 88,
          fontWeight: 800,
          background: 'linear-gradient(135deg, #c084fc 0%, #22d3ee 50%, #fbbf24 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          letterSpacing: '-2px',
          lineHeight: 1.1,
        }}>
          DynaHMRC
        </h1>

        {/* Subtitle */}
        <p style={{
          margin: '16px 0 0',
          fontSize: 26,
          color: 'rgba(255,255,255,0.5)',
          fontWeight: 300,
          letterSpacing: '4px',
          textTransform: 'uppercase',
        }}>
          Decentralized Multi-Robot Collaboration
        </p>

        {/* Divider */}
        <div style={{
          width: 80,
          height: 3,
          background: 'linear-gradient(90deg, #c084fc, #22d3ee)',
          margin: '32px auto',
          borderRadius: 2,
        }} />

        {/* Description */}
        <p style={{
          fontSize: 18,
          color: 'rgba(255,255,255,0.4)',
          maxWidth: 600,
          margin: '0 auto 48px',
          lineHeight: 1.8,
          fontWeight: 300,
        }}>
          Four heterogeneous robots collaborate to complete complex tasks
          <br />through LLM-powered negotiation and planning
        </p>

        {/* Start Button */}
        <button
          onClick={onEnter}
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
          style={{
            padding: '20px 72px',
            fontSize: 22,
            fontWeight: 700,
            border: 'none',
            borderRadius: 60,
            cursor: 'pointer',
            color: '#fff',
            letterSpacing: '2px',
            background: hover
              ? 'linear-gradient(135deg, #7c3aed, #0891b2)'
              : 'linear-gradient(135deg, #8b5cf6, #06b6d4)',
            boxShadow: hover
              ? '0 0 40px rgba(139,92,246,0.5), 0 0 80px rgba(6,182,212,0.3)'
              : '0 0 25px rgba(139,92,246,0.25)',
            transform: hover ? 'translateY(-3px) scale(1.03)' : 'translateY(0)',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            margin: '0 auto',
          }}
        >
          <span style={{ fontSize: 26 }}>✦</span>
          START MISSION
          <span style={{ fontSize: 26 }}>✦</span>
        </button>
      </div>

      {/* Bottom text */}
      <div style={{
        position: 'absolute',
        bottom: 30,
        zIndex: 1,
        fontSize: 14,
        color: 'rgba(255,255,255,0.15)',
        letterSpacing: '2px',
      }}>
        DYNAHMRC v1.0 · Powered by DeepSeek
      </div>
    </div>
  );
}
