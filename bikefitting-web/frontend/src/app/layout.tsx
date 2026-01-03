import type { Metadata } from 'next'
import { Outfit, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const outfit = Outfit({
  subsets: ['latin'],
  variable: '--font-geist',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-geist-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Bike Fitting Analysis | AI-Powered Cycling Optimization',
  description: 'Upload cycling videos for instant AI analysis. Get bike angle detection, joint angle measurements, and personalized fitting recommendations.',
  keywords: ['bike fitting', 'cycling', 'AI', 'pose estimation', 'bike angle'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-surface-950 text-surface-50 antialiased">
        {/* Background grid pattern */}
        <div className="fixed inset-0 bg-grid-pattern bg-grid opacity-50 pointer-events-none" />
        
        {/* Gradient orbs for visual interest */}
        <div className="fixed top-0 left-1/4 w-96 h-96 bg-brand-500/10 rounded-full blur-[128px] pointer-events-none" />
        <div className="fixed bottom-0 right-1/4 w-96 h-96 bg-brand-600/10 rounded-full blur-[128px] pointer-events-none" />
        
        {/* Main content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}

