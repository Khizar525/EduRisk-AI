import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EduRisk AI — Explainable Academic Risk Intelligence",
  description:
    "ML-powered student academic risk prediction with SHAP explainability. Built with FastAPI, Next.js, Scikit-learn, and XGBoost.",
  keywords: [
    "machine learning",
    "academic risk",
    "student prediction",
    "SHAP explainability",
    "FastAPI",
    "Next.js",
    "scikit-learn",
    "XGBoost",
  ],
  authors: [{ name: "EduRisk AI Team" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://edurisk-ai.vercel.app",
    siteName: "EduRisk AI",
    title: "EduRisk AI — Explainable Academic Risk Intelligence",
    description:
      "Predict which students are at academic risk — and explain exactly why. ML pipeline with SHAP explainability, FastAPI REST API, and interactive Next.js dashboard.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "EduRisk AI — Academic Risk Prediction Platform",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "EduRisk AI — Explainable Academic Risk Intelligence",
    description:
      "ML-powered student academic risk prediction with SHAP explainability.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-bg antialiased">
        {children}
      </body>
    </html>
  );
}
