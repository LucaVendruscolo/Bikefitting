import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'BikeFitting AI',
  description: 'AI-powered bike fit analysis - analyze your cycling posture and bike angle',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {/* Animated background */}
        <div className="gradient-bg" />
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        
        {children}
      </body>
    </html>
  );
}

