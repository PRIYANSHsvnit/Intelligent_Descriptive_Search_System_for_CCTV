import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
});
// wide grotesque for headings and labels — the letterforms of road signs and
// number plates, which is what this tool spends its day reading
const archivo = localFont({
  src: "./fonts/ArchivoVF.woff2",
  variable: "--font-display",
  weight: "400 900",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Drishti · CCTV search",
  description: "Describe a person or vehicle in plain words and find every matching clip across cameras.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} ${archivo.variable}`}>
        {children}
      </body>
    </html>
  );
}
