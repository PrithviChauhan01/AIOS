import type { Metadata } from "next";
import { Figtree, DM_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { ThemeProvider } from "@/lib/theme-context";
import { TokenProvider } from "@/lib/token-context";
import { SessionProvider } from "@/lib/session-context";
import Nav from "@/components/Nav";

const figtree = Figtree({
  variable: "--font-figtree",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const dmMono = DM_Mono({
  variable: "--font-dm-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "AIOS",
  description: "Personal AI assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${figtree.variable} ${dmMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="h-screen flex flex-col overflow-hidden bg-bg text-fg">
        {/* Runs before hydration paint so the correct theme class is on <html>
            from the first frame — no flash of the wrong theme. */}
        <Script id="theme-init" strategy="beforeInteractive">
          {`(function(){try{var t=localStorage.getItem('aios_theme');document.documentElement.classList.add(t==='light'?'light':'dark');}catch(e){document.documentElement.classList.add('dark');}})();`}
        </Script>
        <ThemeProvider>
          <TokenProvider>
            <SessionProvider>
              <Nav />
              {children}
            </SessionProvider>
          </TokenProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
