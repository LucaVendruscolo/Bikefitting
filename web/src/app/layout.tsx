import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'BikeFitting - AI Bike Fit Analysis',
  description: 'Upload a video of yourself cycling to get AI-powered bike fit recommendations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-darker text-white min-h-screen">
        {children}
      </body>
    </html>
  );
}

